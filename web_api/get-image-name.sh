DOCKER_IMAGE_NAME="t300/mineru"
PADDLE_MODE="${PADDLE_MODE:-cpu}"
# web_api更新時請更新這邊的日期版本部分
VERSION="0.8.1-paddle-${PADDLE_MODE}-beta"

repo_and_tag="${DOCKER_IMAGE_NAME}:${VERSION}"

echo "${repo_and_tag}"
