# Use the official Ubuntu base image
FROM ubuntu:22.04

# ENV http_proxy http://127.0.0.1:7890
# ENV https_proxy http://127.0.0.1:7890

# Set environment variables to non-interactive to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

# ADD sources.list /etc/apt
# RUN apt-get clean

# Update the package list and install necessary packages
RUN apt-get -q update \
    && apt-get -q install -y --no-install-recommends \
    apt-utils \
    bats \
    build-essential
RUN apt-get update && apt-get install -y tzdata vim net-tools procps lsof curl wget iputils-ping telnet lrzsz git

RUN apt-get update && \
    apt-get install -y \
    software-properties-common && \
    apt-get update && \
    apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3.10-distutils \
    python3-pip \
    wget \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# RUN unset http_proxy && unset https_proxy
# change timezone.
RUN ln -fs /usr/share/zoneinfo/Asia/Taipei /etc/localtime
RUN dpkg-reconfigure -f noninteractive tzdata

# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
# Create a virtual environment for MinerU
RUN python3 -m venv /opt/mineru_venv

RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
    pip install --upgrade pip && \
    pip install fastapi \
    uvicorn \
    python-multipart \
    ultralytics==8.3.39 \
    paddleocr==2.7.3 \
    paddlepaddle==2.5.2 \
    struct-eqtable==0.1.0 \
    unimernet==0.1.6 \
    pypandoc==1.15 \
    torch==2.3.1 \
    pycocotools==2.0.8 \
    wheel==0.45.0 \
    ninja>=1.10.0 \
    magic-pdf==0.8.1"

# install paddlepaddle-gpu package
# 注意: paddlepaddle-gpu 對於顯卡版本十分敏感，若安裝後發現異常，可以先嘗試移除 gpu 版本，改用原先的 cpu 版本
ARG PADDLE_MODE=cpu
RUN if [ "${PADDLE_MODE}" = "gpu" ]; then\
      echo "Installing paddlepaddle-gpu..."; \
      . /opt/mineru_venv/bin/activate && \
      pip install paddlepaddle-gpu==3.0.0b1 --trusted-host www.paddlepaddle.org.cn --trusted-host paddle-whl.bj.bcebos.com \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu118/; \
    fi

RUN /bin/bash -c "source /opt/mineru_venv/bin/activate && \
    pip3 install 'git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02' --no-build-isolation"

# MinerU 在早期版本沒有辦法透過 magic-pdf.json 關閉公式解析功能，因此手動修改 model_configs.yaml
RUN sed -i "s/formula: True/formula: False/g" /opt/mineru_venv/lib/python3.10/site-packages/magic_pdf/resources/model_config/model_configs.yaml

# Copy the configuration file template and set up the model directory
COPY magic-pdf.json /root/magic-pdf.json
ADD app.py /root/app.py

WORKDIR /root

# Set the models directory in the configuration file (adjust the path as needed)
# RUN sed -i 's|/tmp/models|/opt/models|g' /root/magic-pdf.json

# Create the models directory
# RUN mkdir -p /opt/models

# Set the entry point to activate the virtual environment and run the command line tool
# ENTRYPOINT ["/bin/bash", "-c", "source /opt/mineru_venv/bin/activate && exec \"$@\" && python3 app.py", "--"]


# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run FastAPI using Uvicorn, pointing to app.py and binding to 0.0.0.0:8000
CMD ["/bin/bash", "-c", "source /opt/mineru_venv/bin/activate && uvicorn app:app --host 0.0.0.0 --port 8000"]
