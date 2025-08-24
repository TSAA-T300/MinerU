#!/bin/bash -e

PADDLE_MODE="${PADDLE_MODE:-cpu}"
# web_api更新時請更新這邊的日期版本部分
VERSION="0.8.1-paddle-${PADDLE_MODE}-20250824-01"
IMAGE_NAME="t300/mineru:${VERSION}"

echo "building $IMAGE_NAME..."

docker build \
    --build-arg PADDLE_MODE=${PADDLE_MODE} \
    --build-arg IMAGE_NAME=${IMAGE_NAME} \
    --tag ${IMAGE_NAME} \
    --force-rm \
    .
