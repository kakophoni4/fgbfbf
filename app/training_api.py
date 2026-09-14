import json
import time
import uuid
import gzip
import io
from typing import Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import Field, model_validator, ValidationError

from app.db import connect, fetchone, fetchall, audit
from app.schemas import StrictModel, Message, ModelAnswer, Connection, TestRequest
from app.training_store import MODELS


class TrainingRecord(StrictModel):
    chat_id: str = Field(min_length=1, max_length=128)
    system: str = Field(min_length=1, max_length=16000)
    messages: list[Message] = Field(min_length=1, max_length=50)
    answer: ModelAnswer

    @model_validator(mode='after')
    def valid_history(self):
        if self.messages[-1].role != 'user':
            raise ValueError('История должна завершаться сообщением клиента')
        return self


class DatasetCreate(StrictModel):
    name: str = Field(min_length=1, max_length=120)
    records: list[TrainingRecord] = Field(min_length=1, max_length=10000)


class Approval(StrictModel):
    reviewed: Literal[True]
    personal_data_removed: Literal[True]
    authorized_services_only: Literal[True]


class TrainingConfig(StrictModel):
    dataset_id: str
    base_model: Literal['t-tech/T-lite-it-2.1', 't-tech/T-pro-it-2.1']
    epochs: float = Field(default=1, ge=0.1, le=3)
    max_length: int = Field(default=1024, ge=256, le=2048)
    learning_rate: float = Field(default=0.0001, ge=0.000001, le=0.0003)
    seed: int = Field(default=42, ge=0, le=2147483647)


class Activation(StrictModel):
    quality_reviewed: Literal[True]


def build_router(admin, authorized, fail, settings, ollama, submit):
    router = APIRouter(prefix='/v1', dependencies=[Depends(admin)])

    @router.get('/training/status')
    async def status():
        async with connect() as db:
            worker = await fetchone(db, 'SELECT * FROM training_worker WHERE id=1')
        return {'worker_online': bool(worker and time.time()-worker['heartbeat'] < 30),
                'resources': json.loads(worker['resources']) if worker else None,
                'profiles': MODELS, 'automatic_interruption': False}

    @router.post('/datasets', status_code=201)
    async def create_dataset(body: DatasetCreate):
        records = [r.model_dump() for r in body.records]
        # Exact duplicates removed; conflicting answers require human review.
        unique = {json.dumps(r, sort_keys=True, ensure_ascii=False): r for r in records}
        records = list(unique.values())
        identity = uuid.uuid4().hex
        async with connect() as db:
            await db.execute('INSERT INTO datasets(id,name,records,created_at) VALUES(?,?,?,?)',
                             (identity, body.name, json.dumps(records, ensure_ascii=False), time.time()))
            await audit(db, 'dataset.created', identity)
        return {'id': identity, 'count': len(records), 'approved': False,
                'duplicates_removed': len(body.records)-len(records)}

    @router.post('/datasets/import', status_code=201)
    async def import_dataset(request: Request, name: str = Query(min_length=1, max_length=120)):
        raw = await request.body()
        if raw[:2] == b'\x1f\x8b':
            try:
                with gzip.GzipFile(fileobj=io.BytesIO(raw)) as source:
                    raw = source.read(32_000_001)
            except (OSError, EOFError):
                fail(422, 'invalid_gzip', 'Повреждённый gzip')
        if len(raw) > 32_000_000:
            fail(413, 'dataset_too_large', 'Распакованный датасет превышает 32 МБ')
        records = []
        try:
            for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                # Accept the service's approved JSONL export as well as canonical records.
                if isinstance(value, dict) and 'example_id' in value:
                    msgs = value['messages']
                    value = {'chat_id': value['chat_id'], 'system': msgs[0]['content'],
                             'messages': msgs[1:-1], 'answer': json.loads(msgs[-1]['content'])}
                records.append(TrainingRecord.model_validate(value))
                if len(records) > 10000:
                    fail(422, 'too_many_records', 'Максимум 10000 примеров')
        except (UnicodeError, ValueError, KeyError, TypeError, IndexError, ValidationError):
            fail(422, 'invalid_dataset_record', f'Неверный формат строки {locals().get("number", 1)}: нужны chat_id, system, messages и answer')
        if not records:
            fail(422, 'empty_dataset', 'Файл не содержит примеров')
        return await create_dataset(DatasetCreate(name=name, records=records))

    @router.post('/datasets/from-approved', status_code=201)
    async def from_approved(name: str = Query(min_length=1, max_length=120)):
        async with connect() as db:
            rows = await fetchall(db, '''SELECT e.answer_json,r.input_json,r.system_prompt
                FROM examples e JOIN requests r ON r.request_id=e.source_request_id
                WHERE e.approved=1 ORDER BY e.id LIMIT 10001''')
        if not rows or len(rows) > 10000:
            fail(422, 'dataset_size', 'Нужно от 1 до 10000 одобренных примеров')
        records = []
        for row in rows:
            original = json.loads(row['input_json'])
            records.append(TrainingRecord(chat_id=original['chat_id'], system=row['system_prompt'],
                           messages=original['messages'], answer=json.loads(row['answer_json'])))
        return await create_dataset(DatasetCreate(name=name, records=records))

    @router.get('/datasets')
    async def datasets(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        async with connect() as db:
            return {'items': await fetchall(db, 'SELECT id,name,approved,created_at FROM datasets ORDER BY created_at DESC LIMIT ? OFFSET ?', (limit, offset))}

    @router.get('/datasets/{dataset_id}')
    async def dataset(dataset_id: str, limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        async with connect() as db:
            row = await fetchone(db, 'SELECT * FROM datasets WHERE id=?', (dataset_id,))
        if not row:
            fail(404, 'dataset_not_found', 'Датасет не найден')
        records = json.loads(row.pop('records'))
        return {**row, 'count': len(records), 'records': records[offset:offset+limit]}

    @router.post('/datasets/{dataset_id}/approve')
    async def approve(dataset_id: str, body: Approval):
        async with connect() as db:
            cursor = await db.execute('UPDATE datasets SET approved=1 WHERE id=?', (dataset_id,))
            if not cursor.rowcount:
                fail(404, 'dataset_not_found', 'Датасет не найден')
            await audit(db, 'dataset.approved', dataset_id)
        return {'approved': True}

    @router.post('/training/jobs', status_code=202)
    async def train(body: TrainingConfig):
        state = await status()
        if not state['worker_online']:
            fail(503, 'trainer_offline', 'GPU worker не подключён')
        need = MODELS[body.base_model]
        resource = state['resources']
        for field, required in need.items():
            if resource.get(field, 0) < required:
                fail(409, 'insufficient_resources', f'{field}: нужно свободно {required} GiB, доступно {resource.get(field, 0):.1f}')
        config = body.model_dump()
        connection = Connection.model_validate_json((await settings())['connection'])
        if not connection.base_url or not connection.model:
            fail(409, 'connection_not_configured', 'Сначала настройте Ollama')
        config['connection'] = connection.model_dump()
        identity = uuid.uuid4().hex
        async with connect() as db:
            await db.execute('BEGIN IMMEDIATE')
            ds = await fetchone(db, 'SELECT * FROM datasets WHERE id=?', (body.dataset_id,))
            if not ds or not ds['approved']:
                fail(409, 'dataset_not_approved', 'Нужен проверенный и одобренный датасет')
            records = json.loads(ds['records'])
            if len(records) < 20 or len({r['chat_id'] for r in records}) < 5:
                fail(422, 'dataset_too_small', 'Минимум 20 примеров из 5 разных диалогов')
            if await fetchone(db, "SELECT id FROM training_jobs WHERE status IN ('queued','running')"):
                fail(409, 'trainer_busy', 'Предыдущее обучение ещё не завершено')
            await db.execute('INSERT INTO training_jobs(id,dataset_id,config,created_at,updated_at) VALUES(?,?,?,?,?)',
                             (identity, body.dataset_id, json.dumps(config), time.time(), time.time()))
            await audit(db, 'training.queued', identity)
        return {'id': identity, 'status': 'queued'}

    @router.get('/training/jobs')
    async def jobs(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)):
        async with connect() as db:
            return {'items': await fetchall(db, 'SELECT id,dataset_id,status,error,created_at,updated_at FROM training_jobs ORDER BY created_at DESC LIMIT ? OFFSET ?', (limit, offset))}

    @router.get('/training/jobs/{job_id}')
    async def job(job_id: str):
        async with connect() as db:
            row = await fetchone(db, 'SELECT * FROM training_jobs WHERE id=?', (job_id,))
        if not row:
            fail(404, 'job_not_found', 'Задача не найдена')
        row['config'] = json.loads(row['config'])
        row['metrics'] = json.loads(row['metrics']) if row['metrics'] else None
        return row

    @router.get('/training/jobs/{job_id}/logs')
    async def logs(job_id: str, after_id: int = Query(0, ge=0)):
        async with connect() as db:
            rows = await fetchall(db, 'SELECT * FROM training_events WHERE job_id=? AND id>? ORDER BY id LIMIT 200', (job_id, after_id))
        for row in rows:
            row['event'] = json.loads(row['event'])
        return {'items': rows, 'next_after_id': rows[-1]['id'] if rows else after_id}

    @router.post('/training/jobs/{job_id}/cancel')
    async def cancel(job_id: str):
        async with connect() as db:
            cursor = await db.execute("UPDATE training_jobs SET cancel=1 WHERE id=? AND status IN ('queued','running')", (job_id,))
            if not cursor.rowcount:
                fail(409, 'job_not_cancellable', 'Задача отсутствует или уже завершена')
            await audit(db, 'training.cancel_requested', job_id)
        return {'cancel_requested': True}

    @router.get('/model-versions')
    async def versions():
        async with connect() as db:
            rows = await fetchall(db, 'SELECT * FROM model_versions ORDER BY created_at DESC LIMIT 200')
        for row in rows:
            row['metrics'] = json.loads(row['metrics'])
        return {'items': rows}

    async def version(identity):
        async with connect() as db:
            row = await fetchone(db, 'SELECT * FROM model_versions WHERE id=?', (identity,))
        if not row:
            fail(404, 'version_not_found', 'Версия модели не найдена')
        return row

    @router.post('/model-versions/{identity}/test')
    async def test(identity: str, body: TestRequest):
        row = await version(identity)
        current = Connection.model_validate_json((await settings())['connection'])
        if current.base_url != row['endpoint']:
            fail(409, 'endpoint_changed', 'Версия находится на другом Ollama endpoint')
        return await submit(body, test=True, model_override=row['model'])

    @router.post('/model-versions/{identity}/approve')
    async def approve_version(identity: str, body: Activation):
        row = await version(identity)
        async with connect() as db:
            # Require a completed real generation on this candidate, not just a checkbox.
            requests = await fetchall(db, "SELECT response_json FROM requests WHERE status='succeeded'")
            if not any(json.loads(r['response_json']).get('model') == row['model'] for r in requests):
                fail(409, 'test_required', 'Сначала получите тестовый ответ этой модели')
            await db.execute('UPDATE model_versions SET evaluated=1 WHERE id=?', (identity,))
            await audit(db, 'model.approved', identity)
        return {'approved': True}

    @router.post('/model-versions/{identity}/activate')
    async def activate(identity: str):
        row = await version(identity)
        if not row['evaluated']:
            fail(409, 'review_required', 'Проверьте и одобрите новую модель')
        config = Connection.model_validate_json((await settings())['connection'])
        if row['endpoint'] != config.base_url:
            fail(409, 'endpoint_changed', 'Ollama endpoint изменился')
        listing = await ollama(config, 'GET', '/api/tags')
        if row['model'] not in {r.get('name') for r in listing.get('models', [])}:
            fail(409, 'model_missing', 'Модель отсутствует в Ollama')
        async with connect() as db:
            await db.execute('BEGIN IMMEDIATE')
            state = await fetchone(db, 'SELECT connection FROM settings WHERE id=1')
            latest = Connection.model_validate_json(state['connection'])
            if latest.base_url != config.base_url:
                fail(409, 'endpoint_changed', 'Настройки изменились, повторите запрос')
            previous = latest.model
            latest.model = row['model']
            cursor = await db.execute('INSERT INTO model_activations(previous_model,next_model,endpoint,created_at) VALUES(?,?,?,?)',
                                     (previous, latest.model, latest.base_url, time.time()))
            await db.execute('UPDATE settings SET connection=? WHERE id=1', (latest.model_dump_json(),))
            await audit(db, 'model.activated', identity)
        return {'model': latest.model, 'activation_id': cursor.lastrowid, 'previous_model': previous}

    @router.post('/model-activations/{activation_id}/rollback')
    async def rollback(activation_id: int):
        async with connect() as db:
            row = await fetchone(db, 'SELECT * FROM model_activations WHERE id=?', (activation_id,))
        if not row:
            fail(404, 'activation_not_found', 'Активация не найдена')
        config = Connection.model_validate_json((await settings())['connection'])
        listing = await ollama(config, 'GET', '/api/tags')
        names = {r.get('name') for r in listing.get('models', [])}
        if row['previous_model'] not in names and row['previous_model']+':latest' not in names:
            fail(409, 'model_missing', 'Предыдущая модель отсутствует в Ollama')
        async with connect() as db:
            await db.execute('BEGIN IMMEDIATE')
            state = await fetchone(db, 'SELECT connection FROM settings WHERE id=1')
            config = Connection.model_validate_json(state['connection'])
            if config.model != row['next_model'] or config.base_url != row['endpoint']:
                fail(409, 'activation_conflict', 'Активная модель или endpoint уже изменились')
            config.model = row['previous_model']
            await db.execute('UPDATE settings SET connection=? WHERE id=1', (config.model_dump_json(),))
            await audit(db, 'model.rolled_back', activation_id)
        return {'model': config.model}

    return router
