#!/bin/bash -e

PADDLE_MODE="${PADDLE_MODE:-cpu}"
# web_api更新時請更新這邊的日期版本部分
PRJ_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_NAME="$(bash "${PRJ_DIR}/get-image-name.sh")"

echo "building $IMAGE_NAME..."

docker build \
    --build-arg PADDLE_MODE=${PADDLE_MODE} \
    --build-arg IMAGE_NAME=${IMAGE_NAME} \
    --tag ${IMAGE_NAME} \
    --force-rm \
    .
