"""One worker; jobs run in isolated child process. No Ollama stop/unload calls."""
import asyncio
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from app.db import connect, fetchone
from app.training_store import initialize_training, event, MODELS

ROOT = Path('/data/training')


def resources():
    result = {'vram_gib': 0, 'ram_gib': 0, 'disk_gib': shutil.disk_usage('/data').free / 2**30}
    try:
        info = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], timeout=5, text=True)
        # Use GPU 0 only; Docker restricts visible devices to chosen GPU.
        result['vram_gib'] = float(info.splitlines()[0]) / 1024
        result['ram_gib'] = next(int(line.split()[1]) / 1024**2 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    except (OSError, ValueError, subprocess.SubprocessError, StopIteration):
        result['error'] = 'GPU/memory telemetry unavailable'
    return result


async def heartbeat():
    while True:
        data = await asyncio.to_thread(resources)
        async with connect() as db:
            await db.execute('INSERT OR REPLACE INTO training_worker(id,heartbeat,resources) VALUES(1,?,?)',
                             (time.time(), json.dumps(data)))
        await asyncio.sleep(5)


async def finish(identity, status, error=None):
    async with connect() as db:
        await db.execute('UPDATE training_jobs SET status=?,error=?,updated_at=? WHERE id=?',
                         (status, error, time.time(), identity))
    await event(identity, {'stage': status, 'error': error})


async def stop_process(process):
    if process.returncode is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), 15)
        except asyncio.TimeoutError:
            os.killpg(process.pid, signal.SIGKILL)
            await process.wait()


async def execute(job):
    identity = job['id']
    config = json.loads(job['config'])
    available = await asyncio.to_thread(resources)
    for field, need in MODELS[config['base_model']].items():
        if available.get(field, 0) < need:
            await finish(identity, 'failed', f'insufficient_resources: {field}, required={need}, free={available.get(field, 0):.1f}')
            return
    folder = ROOT / identity
    folder.mkdir(parents=True, exist_ok=False)
    async with connect() as db:
        ds = await fetchone(db, 'SELECT records FROM datasets WHERE id=?', (job['dataset_id'],))
    (folder/'input.json').write_text(ds['records'])
    (folder/'config.json').write_text(json.dumps(config))
    await event(identity, {'stage': 'starting', 'resources': available})
    # A private raw log helps server diagnosis; API exposes only structured metrics/events.
    with (folder/'process.log').open('wb') as output:
        process = await asyncio.create_subprocess_exec(sys.executable, '-m', 'trainer.run', str(folder),
            stdout=output, stderr=asyncio.subprocess.STDOUT, start_new_session=True)
        offset = 0
        try:
            while process.returncode is None:
                await asyncio.sleep(2)
                async with connect() as db:
                    state = await fetchone(db, 'SELECT cancel FROM training_jobs WHERE id=?', (identity,))
                if state['cancel']:
                    await stop_process(process)
                    await finish(identity, 'cancelled')
                    return
                events = folder/'events.jsonl'
                if events.exists():
                    with events.open() as log:
                        log.seek(offset)
                        while True:
                            position = log.tell()
                            line = log.readline()
                            if not line:
                                break
                            try:
                                payload = json.loads(line)
                            except ValueError:
                                log.seek(position)
                                break
                            await event(identity, payload)
                        offset = log.tell()
            if process.returncode != 0:
                error_file = folder/'error.json'
                error = json.loads(error_file.read_text())['error'] if error_file.exists() else 'training_process_failed (see private process.log)'
                await finish(identity, 'failed', error)
                return
            result = json.loads((folder/'result.json').read_text())
            async with connect() as db:
                await db.execute('BEGIN IMMEDIATE')
                state = await fetchone(db, 'SELECT cancel FROM training_jobs WHERE id=?', (identity,))
                if state['cancel']:
                    await db.execute("UPDATE training_jobs SET status='cancelled',updated_at=? WHERE id=?", (time.time(), identity))
                    return
                await db.execute('INSERT INTO model_versions(id,job_id,model,endpoint,metrics,created_at) VALUES(?,?,?,?,?,?)',
                                 (identity, identity, result['model'], config['connection']['base_url'], json.dumps(result), time.time()))
                await db.execute("UPDATE training_jobs SET status='succeeded',metrics=?,updated_at=? WHERE id=?",
                                 (json.dumps(result), time.time(), identity))
            await event(identity, {'stage': 'succeeded', 'result': result})
        finally:
            await stop_process(process)


async def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT/'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        await initialize_training()
        async with connect() as db:
            await db.execute("UPDATE training_jobs SET status='failed',error='worker_restarted',updated_at=? WHERE status='running'", (time.time(),))
        beat = asyncio.create_task(heartbeat())
        try:
            while True:
                async with connect() as db:
                    await db.execute('BEGIN IMMEDIATE')
                    await db.execute("UPDATE training_jobs SET status='cancelled',updated_at=? WHERE status='queued' AND cancel=1", (time.time(),))
                    job = await fetchone(db, "SELECT * FROM training_jobs WHERE status='queued' ORDER BY created_at LIMIT 1")
                    if job:
                        await db.execute("UPDATE training_jobs SET status='running',updated_at=? WHERE id=?", (time.time(), job['id']))
                if job:
                    try:
                        await execute(job)
                    except Exception as exc:
                        await finish(job['id'], 'failed', type(exc).__name__)
                else:
                    await asyncio.sleep(2)
        finally:
            beat.cancel()
            await asyncio.gather(beat, return_exceptions=True)


if __name__ == '__main__':
    asyncio.run(main())
