FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/data/hf-cache TOKENIZERS_PARALLELISM=false
WORKDIR /service
COPY requirements.txt ./requirements.txt
COPY trainer/requirements.txt ./training-requirements.txt
RUN pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128 \
    && pip install --no-cache-dir -r requirements.txt -r training-requirements.txt \
    && useradd --uid 10001 --create-home service
COPY app ./app
COPY trainer ./trainer
USER service
CMD ["python", "-m", "trainer.worker"]
