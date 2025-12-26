#!/bin/bash -e

set -euo pipefail
export $(grep -v '^#' .env | xargs -0)
PRJ_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="$(bash "${PRJ_DIR}/get-image-name.sh")"

CONVERTER_MODE="${CONVERTER_MODE:-cuda}"
VRAM_MAX_MB="${VRAM_MAX_MB}"

docker run \
    --name mineru-dev \
    --rm \
    -it \
    --pid=host \
    -p 8888:8000 \
    -v "${PRJ_DIR}/app.py:/root/app.py" \
    -v "${PRJ_DIR}/magic-pdf.json:/root/magic-pdf.json" \
    -v "${HOME_DIR}/.paddleocr:/root/.paddleocr:ro" \
    -v "${HOME_DIR}/hf/PDF-Extract-Kit/models:/opt/models:ro" \
    -v "${HOME_DIR}/plugin-data-tmp:/root/output" \
    -e CONVERTER_MODE="${CONVERTER_MODE}" \
    -e VRAM_MAX_MB="${VRAM_MAX_MB}" \
    --gpus device=2 \
    --entrypoint bash \
    "${IMAGE_NAME}" \
    -c "echo 'source /opt/mineru_venv/bin/activate && sed -i s/\"device-mode\".*/\"device-mode\":\"${CONVERTER_MODE}\",/g /root/magic-pdf.json && exec uvicorn app:app --host 0.0.0.0 --port 8000 --reload' && bash"
