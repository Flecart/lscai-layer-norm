    FROM nvcr.io/nvidia/pytorch:24.11-py3

    # setup
    RUN apt-get update && apt-get install python3-pip python3-venv -y
    RUN apt-get install -y libc-bin

    WORKDIR /app
    COPY pyproject.toml uv.lock ./
    RUN pip install --no-cache-dir uv
    RUN uv sync --frozen --no-dev

    # Install the rest of dependencies.
    RUN uv pip install \
        datasets \
        transformers \
        accelerate \
        wandb \
        dacite \
        pyyaml \
        numpy \ 
        packaging \
        safetensors \
        tqdm \
        sentencepiece \
        tensorboard \
        pandas \
        jupyter \
        deepspeed \
        seaborn

    # RUN pip install -r requirements.txt

    # Create a work directory
    RUN mkdir -p /workspace