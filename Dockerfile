FROM pytorch/pytorch

RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    git \
    vim \
    && rm -rf /var/lib/apt/lists/*

RUN pip install torch tqdm accelerate matplotlib wandb

COPY . .