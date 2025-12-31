#!/bin/bash -e

export $(grep -v '^#' .env | xargs -0)
PRJ_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="$(bash "${PRJ_DIR}/get-image-name.sh")"

CONVERTER_MODE="${CONVERTER_MODE:-cuda}"
DATA_DIR="$(cd ~ && pwd)"

docker run \
    --name mineru-dev \
    --rm \
    -it \
    --pid=host \
    -p 8888:8000 \
    -v "${PRJ_DIR}/app.py:/root/app.py" \
    -v "${PRJ_DIR}/magic-pdf.json:/root/magic-pdf.json" \
    -v "${DATA_DIR}/.paddleocr:/root/.paddleocr:ro" \
    -v "${DATA_DIR}/hf/PDF-Extract-Kit/models:/opt/models:ro" \
    -v "${DATA_DIR}/plugin-data-tmp:/root/output" \
    -e CONVERTER_MODE="${CONVERTER_MODE}" \
    -e VRAM_MAX_MB="${VRAM_MAX_MB}" \
    --gpus device=2 \
    --entrypoint bash \
    "${IMAGE_NAME}" \
    -c "echo $'Welcome to develop mode! To start service, run:\n- source /opt/mineru_venv/bin/activate \n- exec uvicorn app:app --host 0.0.0.0 --port 8000 --reload' && bash"
