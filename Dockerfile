FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.8 python3.8-venv python3-pip git && \
    rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

WORKDIR /opt/parison
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir --upgrade "pip==23.3.2" && \
    python -m pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python -m pip install --no-cache-dir -e .

ENTRYPOINT ["parison"]
CMD ["--help"]
