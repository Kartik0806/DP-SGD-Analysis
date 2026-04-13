python3 -m train_dp --batch-size=32 \
    --num-epochs=10 \
    --task="sst2" \
    --noise-multiplier=0 \
    --learning-rate=5e-4 \
    --max-grad-norm=1 \
    --disable-poisson-sampling \
    --use-wandb \
    --wandb-project="dp-sgd-analysis" \
    --wandb-run-name="data-sweep-clip-1" \
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
