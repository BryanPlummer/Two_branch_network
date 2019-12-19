if [ "$1" = "--train" ]; then
    CUDA_VISIBLE_DEVICES=$2 \
    python train_embedding_nn.py \
    --dataset $3 \
    --language_model $4 \
    --name $5
fi

if [ "$1" = "--test" ]; then
    CUDA_VISIBLE_DEVICES=$2 \
    python eval_embedding_nn.py \
    --dataset $3 \
    --language_model $4 \
    --resume $5 \
    --split test
fi

if [ "$1" = "--val" ]; then
    CUDA_VISIBLE_DEVICES=$2 \
    python eval_embedding_nn.py \
    --dataset $3 \
    --language_model $4 \
    --resume $5 \
    --split val
fi
