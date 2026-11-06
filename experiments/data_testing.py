from torch.utils.data import DataLoader

from latent_forge.data.tinystories import (
    load_tinystories_data,
)

train_ds, val_ds, tokenizer, manifest = load_tinystories_data(
    "data/tinystories_v1",
    seq_len=256,
)

loader = DataLoader(
    train_ds,
    batch_size=4,
    shuffle=True,
    num_workers=0,
)

x, y = next(iter(loader))

print("Input shape:", x.shape)
print("Target shape:", y.shape)
print("Vocabulary:", tokenizer.vocab_size)
print("Training windows:", len(train_ds))
print("Validation windows:", len(val_ds))

print("\nDecoded sample:")
print(tokenizer.decode(x[0].tolist()))