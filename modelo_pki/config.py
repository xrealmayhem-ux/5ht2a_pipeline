"""
Central configuration for the CheMeleon pKi fine-tuning pipeline (5-HT2A).

Active flow: prepare_pki.py (stage curacion) -> train_pki.py (stage modelo_pki)
-> checkpoints/pki/, consumed by generacion/scorer_pki.py via the
generacion/modelos/chemeleon symlink.
"""

from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

# ─── CheMeleon foundation model ─────────────────────────────────────────────
CHEMELEON_URL      = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
CHEMELEON_FILENAME = "chemeleon_mp.pt"
CHEMELEON_PATH     = str(BASE_DIR / CHEMELEON_FILENAME)

# ─── Productive model: seed 2 (best test). [1..5] = unpublished experiment ─
SEED      = 42
PKI_SEEDS = [1, 2, 3, 4, 5]
PRODUCTIVE_SEED = 2

# ─── Hardware ───────────────────────────────────────────────────────────────
ACCELERATOR = "auto"
NUM_WORKERS = 2

# ─── Data splits ────────────────────────────────────────────────────────────
SPLIT_SIZES = (0.7, 0.15, 0.15)

# ─── Architecture ───────────────────────────────────────────────────────────
DROPOUT    = 0.3
BATCH_NORM = False

# ─── Training ───────────────────────────────────────────────────────────────
BATCH_SIZE        = 64
MAX_EPOCHS        = 50
PATIENCE          = 10
GRADIENT_CLIP_VAL = 5.0

# ─── Optimizer ──────────────────────────────────────────────────────────────
WEIGHT_DECAY = 1e-5

# ─── LR scheduler ───────────────────────────────────────────────────────────
SCHEDULER_FACTOR   = 0.5
SCHEDULER_PATIENCE = 5
SCHEDULER_MIN_LR   = 1e-6
