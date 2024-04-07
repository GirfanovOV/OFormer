# OFormers

Transformers unable to scale on long sequences.
This repo is dedicated to explore alternatives to softmax attention and find **O**ptimal trans**Former** achitectures for processing long sequeneces.

Uses Huggingface `accelerate` to handle distributed training.

# Docker
```bash
docker build -t oformer .
docker run --gpus all --shm-size=24gb -it --rm oformer
```

### Inside container

```bash
accelerate config
accelerate launch train.py (args...)
```

# TODOs

- Setup DeepSpeed (ZERO)
- Start experiments with linear attention