#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -f .env ]; then
  echo 'Сначала создайте .env по .env.example и задайте два разных ключа API.' >&2
  exit 1
fi
docker compose up -d --build ai-api
docker compose ps
echo 'Развёртывание запрошено. Проверки выполняются отдельно по docs/SERVER.md.'
