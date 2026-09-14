"""Training metadata shared by API and a single optional worker."""
import json
import time
from app.db import connect


async def initialize_training():
    async with connect() as db:
        await db.executescript('''
        CREATE TABLE IF NOT EXISTS datasets (
          id TEXT PRIMARY KEY, name TEXT NOT NULL, records TEXT NOT NULL,
          approved INTEGER NOT NULL DEFAULT 0, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS training_jobs (
          id TEXT PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id),
          config TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'queued',
          cancel INTEGER NOT NULL DEFAULT 0, metrics TEXT, error TEXT,
          created_at REAL NOT NULL, updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS training_events (
          id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
          event TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_versions (
          id TEXT PRIMARY KEY, job_id TEXT NOT NULL, model TEXT NOT NULL,
          endpoint TEXT NOT NULL, evaluated INTEGER NOT NULL DEFAULT 0,
          metrics TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS model_activations (
          id INTEGER PRIMARY KEY AUTOINCREMENT, previous_model TEXT NOT NULL,
          next_model TEXT NOT NULL, endpoint TEXT NOT NULL, created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS training_worker (
          id INTEGER PRIMARY KEY CHECK(id=1), heartbeat REAL NOT NULL, resources TEXT NOT NULL
        );
        ''')


async def event(job_id, payload):
    async with connect() as db:
        await db.execute('INSERT INTO training_events(job_id,event,created_at) VALUES(?,?,?)',
                         (job_id, json.dumps(payload, ensure_ascii=False), time.time()))


MODELS = {
    't-tech/T-lite-it-2.1': {'vram_gib': 12, 'ram_gib': 40, 'disk_gib': 80},
    't-tech/T-pro-it-2.1': {'vram_gib': 32, 'ram_gib': 100, 'disk_gib': 260},
}
