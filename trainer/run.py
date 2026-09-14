"""Real QLoRA training, held-out loss, CPU merge, Ollama safetensors import.

No arbitrary executable, filesystem path or remote model code from the HTTP request.
"""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import httpx


def main(folder):
    import torch
    from datasets import Dataset
    from huggingface_hub import HfApi
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
                              DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments,
                              set_seed)

    config = json.loads((folder/'config.json').read_text())
    rows = json.loads((folder/'input.json').read_text())

    def emit(**fields):
        with (folder/'events.jsonl').open('a') as output:
            output.write(json.dumps({'time': time.time(), **fields}, ensure_ascii=False) + '\n')

    set_seed(config['seed'])
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    emit(stage='preparing_data')
    unique = {}
    for row in rows:
        key = json.dumps({k: v for k, v in row.items() if k != 'chat_id'}, sort_keys=True)
        unique.setdefault(key, row)
    rows = list(unique.values())
    groups = sorted({r['chat_id'] for r in rows})
    if len(groups) < 5 or len(rows) < 20:
        raise RuntimeError('Not enough unique examples after deduplication')
    random.Random(config['seed']).shuffle(groups)
    heldout = set(groups[:max(1, len(groups)//5)])
    revision = HfApi().model_info(config['base_model']).sha
    if not revision:
        raise RuntimeError('Cannot record exact base model revision')
    tokenizer = AutoTokenizer.from_pretrained(config['base_model'], revision=revision, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(row):
        messages = [{'role': 'system', 'content': row['system']}] + row['messages']
        prefix = tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False)
        completion = tokenizer.encode(json.dumps(row['answer'], ensure_ascii=False), add_special_tokens=False)
        completion += [tokenizer.eos_token_id]
        ids = prefix + completion
        if len(ids) > config['max_length']:
            raise RuntimeError('Example exceeds max_length; shorten history or increase training max_length')
        return {'input_ids': ids, 'attention_mask': [1]*len(ids),
                'labels': [-100]*len(prefix) + completion}

    train = [tokenize(r) for r in rows if r['chat_id'] not in heldout]
    valid = [tokenize(r) for r in rows if r['chat_id'] in heldout]
    if not train or not valid:
        raise RuntimeError('Empty train/evaluation split')
    (folder/'split.json').write_text(json.dumps({'heldout_chat_ids': sorted(heldout),
        'train_examples': len(train), 'eval_examples': len(valid)}))
    emit(stage='loading_base', train_examples=len(train), eval_examples=len(valid))
    model = AutoModelForCausalLM.from_pretrained(config['base_model'], revision=revision, trust_remote_code=False,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
        device_map={'': 0}, torch_dtype=torch.bfloat16, attn_implementation='sdpa')
    model.config.use_cache = False
    # Merge must use this exact revision, not a later upstream update.
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'], task_type='CAUSAL_LM'))

    class Progress(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            fields = {k: v for k, v in (logs or {}).items() if isinstance(v, (int, float)) and math.isfinite(v)}
            emit(stage='training', step=state.global_step, total_steps=state.max_steps, metrics=fields)

    training_args = TrainingArguments(output_dir=str(folder/'checkpoints'),
        num_train_epochs=config['epochs'], per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=config['learning_rate'], bf16=True, fp16=False,
        gradient_checkpointing=True, optim='adamw_torch', logging_steps=1,
        save_strategy='no', report_to=[], seed=config['seed'], disable_tqdm=True,
        dataloader_num_workers=0)
    trainer = Trainer(model=model, args=training_args, train_dataset=Dataset.from_list(train),
        eval_dataset=Dataset.from_list(valid), callbacks=[Progress()],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100))
    emit(stage='evaluating_baseline')
    before = trainer.evaluate()['eval_loss']
    trainer.train()
    emit(stage='evaluating_candidate')
    after = trainer.evaluate()['eval_loss']
    if not math.isfinite(before) or not math.isfinite(after):
        raise RuntimeError('Non-finite evaluation loss')
    adapter = folder/'adapter'
    trainer.save_model(str(adapter))
    tokenizer.save_pretrained(adapter)
    emit(stage='adapter_saved', baseline_loss=before, candidate_loss=after)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()

    emit(stage='merging_on_cpu')
    base = AutoModelForCausalLM.from_pretrained(config['base_model'], revision=revision,
        trust_remote_code=False, torch_dtype=torch.float16, device_map={'': 'cpu'},
        low_cpu_mem_usage=True)
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload(safe_merge=True)
    target = folder/'merged'
    merged.save_pretrained(target, safe_serialization=True, max_shard_size='2GB')
    tokenizer.save_pretrained(target)
    del merged, base
    gc.collect()
    emit(stage='uploading_to_ollama')
    headers = {}
    if os.getenv('OLLAMA_API_KEY'):
        headers['Authorization'] = 'Bearer ' + os.environ['OLLAMA_API_KEY']
    files = {}
    endpoint = config['connection']['base_url']
    model_name = 'fgbfbf-' + folder.name + ':q4_k_m'
    with httpx.Client(timeout=httpx.Timeout(3600, connect=10), trust_env=False,
                      follow_redirects=False, headers=headers) as client:
        for path in sorted(target.iterdir()):
            if path.suffix not in {'.json', '.safetensors', '.model'}:
                continue
            digest = hashlib.sha256()
            with path.open('rb') as source:
                for chunk in iter(lambda: source.read(4*1024*1024), b''):
                    digest.update(chunk)
            sha = 'sha256:' + digest.hexdigest()
            with path.open('rb') as source:
                response = client.post(endpoint+'/api/blobs/'+sha,
                    content=iter(lambda: source.read(4*1024*1024), b''),
                    headers={'Content-Length': str(path.stat().st_size), 'Content-Type': 'application/octet-stream'})
                response.raise_for_status()
            files[path.name] = sha
            emit(stage='uploading_to_ollama', file=path.name)
        response = client.post(endpoint+'/api/create', json={'model': model_name,
            'files': files, 'quantize': 'q4_K_M', 'stream': False})
        response.raise_for_status()
        if response.json().get('status') != 'success':
            raise RuntimeError('Ollama import did not report success')
        listing = client.get(endpoint+'/api/tags')
        listing.raise_for_status()
        if model_name not in {m.get('name') for m in listing.json().get('models', [])}:
            raise RuntimeError('Imported model not found in Ollama')
    result = {'model': model_name, 'base_model': config['base_model'], 'base_revision': revision,
        'baseline_loss': before, 'candidate_loss': after, 'train_examples': len(train),
        'eval_examples': len(valid), 'quality_approved': False}
    (folder/'result.json').write_text(json.dumps(result))
    emit(stage='exported', **result)


if __name__ == '__main__':
    folder = Path(sys.argv[1])
    try:
        main(folder)
    except Exception as error:
        # Do not expose raw model text, endpoint credentials or dataset contents via API logs.
        safe = str(error) if type(error) is RuntimeError and not str(error).startswith('CUDA') else type(error).__name__
        if 'out of memory' in str(error).lower():
            safe = 'GPU out of memory; no model was activated'
        (folder/'error.json').write_text(json.dumps({'error': safe[:500]}))
        raise
