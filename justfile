# Add, commit and push
acp message="Update":
    git add .
    git commit -m "{{message}}"
    git push


# Default attention mechanisms
attentions := "mha gqa mqa mla"

# Shared experiment configuration
steps := "1000"
batch := "4"
accum := "8"
seq_len := "256"
d_model := "384"
layers := "8"
heads := "8"
d_ff := "1024"
lr := "0.0003"
warmup := "20"
seed := "42"

# Run one attention experiment
train attention:
    uv run python -m latent_forge.training.train_tinystories \
        --attention {{attention}} \
        --mode train \
        --data-dir data/tinystories_v1 \
        --steps {{steps}} \
        --batch-size {{batch}} \
        --accum-steps {{accum}} \
        --seq-len {{seq_len}} \
        --d-model {{d_model}} \
        --layers {{layers}} \
        --heads {{heads}} \
        --d-ff {{d_ff}} \
        --lr {{lr}} \
        --warmup-steps {{warmup}} \
        --eval-every 100 \
        --val-batches 64 \
        --seed {{seed}}

all:
    powershell.exe -NoProfile -ExecutionPolicy Bypass -File run_all.ps1