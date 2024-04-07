FROM pytorch/pytorch

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN conda install -c nvidia cuda-compiler

RUN pip install torch tqdm accelerate matplotlib wandb deepspeed

COPY . .