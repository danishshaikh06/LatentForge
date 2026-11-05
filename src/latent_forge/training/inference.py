import argparse
import json
from pathlib import Path

import torch

from latent_forge.core.paths import BEST_MODEL_FILE, TOKENIZER_FILE, TRAINING_CONFIG_FILE
from latent_forge.tokenization.bpe_tokenizer import BPETokenizer
from latent_forge.models.decoder import (
    build_transformer_from_config,
)


SYSTEM_PROMPT = (
    "User: {prompt}\n"
    "Assistant:\n"
)


def load_tokenizer(path: Path = TOKENIZER_FILE) -> BPETokenizer:
    tokenizer = BPETokenizer()
    tokenizer.load(str(path))
    return tokenizer


def load_training_config(path: Path = TRAINING_CONFIG_FILE) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_prompt(user_prompt: str) -> str:
    return SYSTEM_PROMPT.format(prompt=user_prompt.strip())


def _filter_special_tokens(token_ids, special_token_ids):
    return [token for token in token_ids if token not in special_token_ids]


def _sample_next_token(logits: torch.Tensor, temperature: float, top_k: int | None) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits).item())

    scaled_logits = logits / temperature

    if top_k is not None and top_k > 0:
        top_k = min(top_k, scaled_logits.size(-1))
        values, indices = torch.topk(scaled_logits, top_k)
        filtered = torch.full_like(scaled_logits, float("-inf"))
        filtered.scatter_(0, indices, values)
        scaled_logits = filtered

    probs = torch.softmax(scaled_logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def generate_completion(
    model,
    tokenizer,
    prompt: str,
    seq_len: int,
    pad_id: int,
    eos_id: int,
    device,
    bos_id: int | None = None,
    max_new_tokens: int = 160,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    formatted_prompt = format_prompt(prompt)
    prompt_tokens = tokenizer.encode(formatted_prompt)
    stop_strings = ["\nUser:", "\nAssistant:"]

    generated = []
    if bos_id is not None:
        generated.append(bos_id)
    generated.extend(prompt_tokens)

    model.eval()

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated[-seq_len:]
            x = torch.tensor([context], dtype=torch.long, device=device)
            tgt_mask = model.create_tgt_mask(x, pad_id)
            logits = model(x, tgt_mask)[0, -1]
            next_token = _sample_next_token(logits, temperature=temperature, top_k=top_k)

            if next_token == eos_id:
                break

            generated.append(next_token)

    new_tokens = generated[len(prompt_tokens) + (1 if bos_id is not None else 0) :]
    decoded = tokenizer.decode(_filter_special_tokens(new_tokens, {pad_id, eos_id, bos_id}))

    for stop_string in stop_strings:
        stop_index = decoded.find(stop_string)
        if stop_index != -1:
            decoded = decoded[:stop_index]

    return decoded.strip()


def load_runtime(
    checkpoint_path: Path = BEST_MODEL_FILE,
    config_path: Path = TRAINING_CONFIG_FILE,
    tokenizer_path: Path = TOKENIZER_FILE,
):
    config = load_training_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_transformer_from_config(
        config,
        tgt_vocab_size=config["vocab_size"],
    ).to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    tokenizer = load_tokenizer(tokenizer_path)

    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "tokenizer_path": str(tokenizer_path),
    }


def run_inference(
    prompt: str,
    checkpoint_path: Path = BEST_MODEL_FILE,
    config_path: Path = TRAINING_CONFIG_FILE,
    tokenizer_path: Path = TOKENIZER_FILE,
    max_new_tokens: int = 160,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    runtime = load_runtime(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        tokenizer_path=tokenizer_path,
    )
    config = runtime["config"]

    return generate_completion(
        model=runtime["model"],
        tokenizer=runtime["tokenizer"],
        prompt=prompt,
        seq_len=config["seq_len"],
        pad_id=config["pad_id"],
        eos_id=config["eos_id"],
        bos_id=config.get("bos_id"),
        device=runtime["device"],
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
    )


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run local inference for the C++ code generator.")
    parser.add_argument("prompt", help="Prompt to send to the model.")
    parser.add_argument("--checkpoint", default=str(BEST_MODEL_FILE), help="Path to the checkpoint file.")
    parser.add_argument("--config", default=str(TRAINING_CONFIG_FILE), help="Path to the saved training config.")
    parser.add_argument("--tokenizer", default=str(TOKENIZER_FILE), help="Path to tokenizer.json.")
    parser.add_argument("--max-new-tokens", type=int, default=160, help="Maximum number of generated tokens.")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature. Use 0 for greedy decoding.")
    parser.add_argument("--top-k", type=int, default=40, help="Top-k sampling cutoff.")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    text = run_inference(
        prompt=args.prompt,
        checkpoint_path=Path(args.checkpoint),
        config_path=Path(args.config),
        tokenizer_path=Path(args.tokenizer),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(text)
