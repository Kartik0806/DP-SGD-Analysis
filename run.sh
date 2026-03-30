python3 -m train_dp --batch-size=2048 \
    --num-epochs=1 \
    --task="sst2" \
    --noise-multiplier=0.4 \
    --learning-rate=5e-4 \
    --max-grad-norm=0.1 \
    --disable-poisson-sampling \
    --wandb-project="dp-sgd-analysis" \
    --wandb-run-name="dp-sgd-no-update" \
    --use-wandb \
    --max-physical-batch-size=32 \
    --run-eval \

    # --use-lora

# python3 -m train --batch-size=16 \
#     --num-epochs=5 \
#     --task="sst2" \
#     --wandb-project="dp-sgd-analysis" \
#     --wandb-run-name="no-opacus" \
#     --use-wandb \
#     --grad-accumulation-steps=2 \
#     --learning-rate=2e-5 \
