import os
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite

PATH = os.getenv("DATABASE_PATH", "/data/ai.sqlite3")


@asynccontextmanager
async def connect():
    async with aiosqlite.connect(PATH, timeout=15) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        yield db
        await db.commit()


async def initialize():
    Path(PATH).parent.mkdir(parents=True, exist_ok=True)
    async with connect() as db:
        await db.execute("PRAGMA journal_mode=WAL")
        await db.executescript("""
        CREATE TABLE IF NOT EXISTS settings (
          id INTEGER PRIMARY KEY CHECK(id=1), connection TEXT NOT NULL DEFAULT '{}',
          catalog TEXT NOT NULL DEFAULT '', catalog_version INTEGER NOT NULL DEFAULT 0,
          active_prompt_id INTEGER REFERENCES prompts(id)
        );
        CREATE TABLE IF NOT EXISTS prompts (
          id INTEGER PRIMARY KEY AUTOINCREMENT, title TEXT NOT NULL, content TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
        INSERT OR IGNORE INTO settings(id) VALUES(1);
        CREATE TABLE IF NOT EXISTS requests (
          request_id TEXT PRIMARY KEY, input_hash TEXT NOT NULL, input_json TEXT NOT NULL,
          prompt_id INTEGER NOT NULL REFERENCES prompts(id), catalog_version INTEGER NOT NULL,
          system_prompt TEXT NOT NULL, connection_json TEXT NOT NULL,
          status TEXT NOT NULL, response_json TEXT, error_code TEXT, error_message TEXT,
          created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          finished_at TEXT
        );
        CREATE TABLE IF NOT EXISTS examples (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          source_request_id TEXT NOT NULL UNIQUE REFERENCES requests(request_id),
          answer_json TEXT NOT NULL, note TEXT NOT NULL DEFAULT '',
          approved INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
          updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
        CREATE TABLE IF NOT EXISTS audit (
          id INTEGER PRIMARY KEY AUTOINCREMENT, event TEXT NOT NULL,
          target TEXT NOT NULL,
          created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
        );
        """)
        await db.execute("""UPDATE requests SET status='failed', error_code='service_restarted',
          error_message='Сервис перезапущен. Создайте новый request_id для повторной попытки.',
          finished_at=strftime('%Y-%m-%dT%H:%M:%fZ','now') WHERE status='running'""")


async def fetchone(db, query, args=()):
    async with db.execute(query, args) as cursor:
        row = await cursor.fetchone()
        return dict(row) if row else None


async def fetchall(db, query, args=()):
    async with db.execute(query, args) as cursor:
        return [dict(row) for row in await cursor.fetchall()]


async def audit(db, event, target=""):
    await db.execute("INSERT INTO audit(event,target) VALUES(?,?)", (event, str(target)))
