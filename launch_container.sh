gpu=$1
WANDB_API_KEY=$(cat ./dev/wandb_key)

docker run \
    --gpus device=$gpu \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -v $(pwd):/home/duser/jax_bnn \
    --name jax_bnn\_$gpu \
    --user $(id -u) \
    -it jax_bnn bash