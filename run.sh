python3 -m train_dp --batch-size=2048 \
    --num-epochs=10 \
    --task="sst2" \
    --noise-multiplier=0 \
    --learning-rate=1e-3 \
    --max-grad-norm=1000000 \
    --disable-poisson-sampling \
    --wandb-project="dp-sgd-analysis" \
    --wandb-run-name="yes-update-no-noise-high-lr" \
    --max-physical-batch-size=32 \
    --run-eval \
    --update-weights \


# python3 -m train --batch-size=16 \
#     --num-epochs=5 \
#     --task="sst2" \
#     --wandb-project="dp-sgd-analysis" \
#     --wandb-run-name="no-opacus" \
#     --use-wandb \
#     --grad-accumulation-steps=2 \
#     --learning-rate=2e-5 \
