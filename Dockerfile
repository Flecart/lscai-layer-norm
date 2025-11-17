FROM nvcr.io/nvidia/pytorch:24.11-py3

# setup
RUN apt-get update && apt-get install python3-pip python3-venv -y
RUN apt-get install -y libc-bin

RUN pip install --no-cache-dir uv

# Create a work directory
RUN mkdir -p /workspace