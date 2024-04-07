accelerate launch train.py \
        --num_epoch 120 \
        --batch_size 128 \
        --transformer_num_layers 8 \
        --transformer_model_dim 512 \
        --transformer_num_heads 8 \
        --transformer_ff_dim 2048