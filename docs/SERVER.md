# Развёртывание на Linux-сервере

Команды ниже выполняются на сервере, где будет работать AI API. Нужны git, Docker Engine
и Docker Compose plugin. Установка Ollama отдельная; этот Compose не изменяет существующую Ollama.

## 1. Скачать сервис и создать ключи

```bash
git clone https://github.com/kakophoni4/fgbfbf.git /opt/fgbfbf
cd /opt/fgbfbf
umask 077
cp .env.example .env
python3 - <<'PY'
from pathlib import Path
import secrets
p = Path('.env')
s = p.read_text()
s = s.replace('AI_ADMIN_KEY=\n', 'AI_ADMIN_KEY=' + secrets.token_urlsafe(48) + '\n')
s = s.replace('AI_REPLY_KEY=\n', 'AI_REPLY_KEY=' + secrets.token_urlsafe(48) + '\n')
p.write_text(s)
PY
```

Ключи находятся в /opt/fgbfbf/.env. Передать их команде CRM по защищённому каналу;
не присылать в общие чаты и не сохранять в Git. Повторно не выполняйте cp поверх рабочего .env.

## 2. Выбрать сетевое подключение

По умолчанию сервис доступен только на хосте: http://127.0.0.1:8088.
Для CRM на другом сервере задайте AI_BIND_HOST приватным/VPN IP сервера API
и разрешите вход только от backend CRM. Если нет приватной сети — используйте HTTPS
reverse proxy с ограничением доступа. Не публикуйте ключи и API по открытому HTTP в интернете.

127.0.0.1 внутри контейнера — сам контейнер, а не хост и не CRM.
Если Ollama на Linux-хосте, используйте base_url=http://host.docker.internal:11434,
но Ollama должна слушать адрес, доступный Docker bridge. Ollama, слушающая только 127.0.0.1
хоста, из контейнера недоступна. Изменение её bind-адреса согласуется отдельно:
этот сервис сам настройки Ollama не меняет и ничего не перезапускает.
Если Ollama на другой машине — используйте её приватный IP или защищённый шлюз.

## 3. Собрать и поднять только API

```bash
cd /opt/fgbfbf
bash scripts/deploy.sh
docker compose logs --tail=80 ai-api
curl --fail-with-body http://127.0.0.1:8088/health
```

При другом AI_BIND_HOST замените адрес curl. Health проверяет API/базу, а не Ollama.

## 4. Серверная проверка до подключения клиентов

В том же терминале сервера:

```bash
set -a
. ./.env
set +a
API="http://127.0.0.1:8088"

curl --fail-with-body "$API/v1/capabilities" \
  -H "Authorization: Bearer $AI_REPLY_KEY"

curl --fail-with-body "$API/v1/connection" \
  -H "Authorization: Bearer $AI_ADMIN_KEY"
```

Ожидается training_available=false и configured=false на чистой базе.
Проверить разграничение прав: запрос к /v1/connection с reply-ключом должен вернуть 401.

Задать реальный адрес Ollama; ниже пример для приватной сети, замените IP и имя модели:

```bash
curl --fail-with-body -X PUT "$API/v1/connection" \
  -H "Authorization: Bearer $AI_ADMIN_KEY" -H 'Content-Type: application/json' \
  -d '{"base_url":"http://10.0.0.10:11434","model":"hf.co/t-tech/T-pro-it-2.1-GGUF:Q4_K_M","timeout_seconds":120,"num_ctx":8192,"num_predict":512,"temperature":0.4}'

curl --fail-with-body -X POST "$API/v1/connection/check" \
  -H "Authorization: Bearer $AI_ADMIN_KEY"
```

Создать промпт из файла; id берётся из ответа, а не предполагается равным 1:

```bash
python3 - <<'PY' > /tmp/fgbfbf-prompt.json
from pathlib import Path
import json
print(json.dumps({'title':'Начальная версия','content':Path('prompts/initial.txt').read_text()}))
PY

PROMPT_ID=$(curl --fail-with-body -X POST "$API/v1/prompts" \
  -H "Authorization: Bearer $AI_ADMIN_KEY" -H 'Content-Type: application/json' \
  --data-binary @/tmp/fgbfbf-prompt.json | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')

curl --fail-with-body -X POST "$API/v1/prompts/$PROMPT_ID/activate" \
  -H "Authorization: Bearer $AI_ADMIN_KEY"

curl --fail-with-body -X POST "$API/v1/test/replies" \
  -H "Authorization: Bearer $AI_ADMIN_KEY" -H 'Content-Type: application/json' \
  -d '{"request_id":"server-smoke-0001","chat_id":"test-only","messages":[{"role":"user","content":"Здравствуйте, какие услуги доступны?"}]}'
```

Без каталога корректный ответ не должен выдумывать услуги. Это ручная проверка поведения модели.
Если HTTP 202, получить результат:

```bash
curl --fail-with-body "$API/v1/requests/server-smoke-0001" \
  -H "Authorization: Bearer $AI_ADMIN_KEY"
```

Затем на сервере проверить:

1. Повтор точного запроса возвращает тот же результат без новой генерации.
2. Другой текст с тем же id возвращает 409.
3. Новая версия промпта не активна до явной активации; можно протестировать её через prompt_id.
4. Создание примера → одобрение → экспорт; редактирование исключает его из экспорта до нового одобрения.
5. POST /v1/training/jobs возвращает 501 и не создаёт фоновую задачу.
6. После docker compose restart ai-api настройки, версии и примеры сохранены.
7. При недоступной Ollama ошибка отображается как ошибка, а не ответ клиенту.

Используйте новые request_id для независимых проверок. Реальные чаты не подключать до приёмки.

## Обновление

```bash
cd /opt/fgbfbf
git pull --ff-only origin main
bash scripts/deploy.sh
```

Перезапускается только AI API, существующие сервисы CRM/Ollama не затрагиваются.
При обновлении незавершённые API-запросы могут завершиться ошибкой service_restarted;
планируйте обновление, учитывая активные обращения. Обучение ответы автоматически не останавливает.

## Резервная копия

Не копируйте только sqlite-файл работающей базы без WAL. Используйте SQLite backup API:

```bash
docker compose exec -T ai-api python -c "import sqlite3; src=sqlite3.connect('/data/ai.sqlite3'); dst=sqlite3.connect('/data/backup.sqlite3'); src.backup(dst); dst.close(); src.close()"
docker compose cp ai-api:/data/backup.sqlite3 ./backup.sqlite3
```

Копия содержит переписки: хранить вне Git с ограниченным доступом.
