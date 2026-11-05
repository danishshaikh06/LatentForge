from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR =  Path(__file__).resolve().parents[1] / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"

TOKENIZER_FILE = PROCESSED_DATA_DIR / "tokenizer.json"
TOKENS_FILE = PROCESSED_DATA_DIR / "tokens.npy"
TOKENS_WITH_SPECIAL_FILE = PROCESSED_DATA_DIR / "tokens_with_specialv2.npy"

TRAINING_CONFIG_FILE = ARTIFACTS_DIR / "training_config.json"
BASE_BEST_MODEL_FILE = ARTIFACTS_DIR / "best_model.pt"
BASE_LAST_MODEL_FILE = ARTIFACTS_DIR / "model_epoch_10.pt"
FINETUNE_DIR = ARTIFACTS_DIR / "finetune"
FINETUNE_CONFIG_FILE = FINETUNE_DIR / "finetune_config.json"
FINETUNE_BEST_MODEL_FILE = FINETUNE_DIR / "finetune_best_model.pt"


def get_default_checkpoint_file() -> Path:
    if FINETUNE_BEST_MODEL_FILE.exists():
        return FINETUNE_BEST_MODEL_FILE
    if BASE_BEST_MODEL_FILE.exists():
        return BASE_BEST_MODEL_FILE
    return BASE_LAST_MODEL_FILE


BEST_MODEL_FILE = get_default_checkpoint_file()


def ensure_artifacts_dir() -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    return ARTIFACTS_DIR
