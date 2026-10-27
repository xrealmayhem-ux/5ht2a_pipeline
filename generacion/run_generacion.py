
"""
Staged learning

De Novo generation of molecules for 5-HT2A receptor using Reinvent 4 and Chemprop as scoring function.
"""

import random
import shutil
import subprocess
import sys
import os
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ////////////////////////////////////////////////////////////////////////////
# Core Configuration & Hyperparameters
# ////////////////////////////////////////////////////////////////////////////
SEED = int(os.getenv("SEED", 42))
DEVICE = os.getenv("DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 256))
S1_STEPS = int(os.getenv("S1_STEPS", 500))
TL_EPOCHS = int(os.getenv("TL_EPOCHS", 20))
S2_STEPS = int(os.getenv("S2_STEPS", 1000))
INCEPTION_SMILES_FILE = os.getenv("INCEPTION_SMILES_FILE", "").strip()
USE_CHECKPOINT = os.getenv("USE_CHECKPOINT", "false").strip().lower() in {"1", "true", "yes"}

# Reproducibility for the Python-side data preparation.
random.seed(SEED)

# Resolve the Chemprop Python executable once, but fail early if the environment
# is missing instead of discovering the problem after RL has started.
_chemprop_probe = subprocess.run(
    ["conda", "run", "-n", "modelo_pki", "which", "python"],
    capture_output=True, text=True
)
CHEMPROP_PYTHON_PATH = _chemprop_probe.stdout.strip()
if _chemprop_probe.returncode != 0 or not CHEMPROP_PYTHON_PATH:
    raise RuntimeError(
        "Could not resolve the Python executable in the 'modelo_pki' conda environment. "
        f"stderr: {_chemprop_probe.stderr.strip()}"
    )
#////////////////////////////////////////////////////////////////////////////

OUT_DIR = BASE_DIR / os.getenv("OUT_DIR", "resultados")
OUT_DIR.mkdir(exist_ok=True)

def configured_path(name, default):
    path = Path(os.getenv(name, default))
    return path if path.is_absolute() else BASE_DIR / path


PRIOR_FILE = configured_path("PRIOR_FILE", "modelos/reinvent.prior")
TL_SMILES_FILE = configured_path("TL_SMILES_FILE", "datos/tl_smiles.smi")
SCORER_SCRIPT = configured_path("SCORER_SCRIPT", "scorer_pki.py")

def log(msg):
    print(f"  {msg}", flush=True)

def banner(title):
    n = len(title)
    print("", flush=True)
    print("=" * (n + 8), flush=True)
    print(f"==  {title}  ==", flush=True)
    print("=" * (n + 8), flush=True)
    print("", flush=True)

def run_step(name, cmd, check_file=None, desc=None):
    if desc:
        log(f"Goal: {desc}")
    log(f"Running: {' '.join(cmd)}")
    log(f"Starting: {name}")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    if result.returncode != 0:
        log(f"ERROR in {name} (after {elapsed:.1f}s)")
        stderr = result.stderr.strip()
        if stderr:
            log(f"stderr (last 1000 chars):\n{stderr[-1000:]}")
        sys.exit(1)
    if check_file and not Path(check_file).exists():
        log(f"ERROR: {name} did not generate {check_file}")
        sys.exit(1)
    log(f"OK: {name} completed in {elapsed:.1f}s")
    return result

def resolve_optional_path(path_text: str) -> str:
    """Return an absolute path for an optional file, or empty string."""
    if not path_text:
        return ""
    p = Path(path_text)
    if not p.is_absolute():
        p = BASE_DIR / p
    return str(p.resolve())


def validate_inputs():
    """Fail early on missing files or invalid configuration inputs."""
    required_files = {
        "prior_file": PRIOR_FILE,
        "tl_smiles_file": TL_SMILES_FILE,
        "scorer_script": SCORER_SCRIPT,
    }
    missing = [f"{name}={path}" for name, path in required_files.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required inputs: " + "; ".join(missing))

    if not 1 <= BATCH_SIZE:
        raise ValueError("BATCH_SIZE must be >= 1")
    if not 1 <= S1_STEPS or not 1 <= S2_STEPS or not 1 <= TL_EPOCHS:
        raise ValueError("S1_STEPS, S2_STEPS and TL_EPOCHS must all be >= 1")

    if INCEPTION_SMILES_FILE:
        inception_path = Path(resolve_optional_path(INCEPTION_SMILES_FILE))
        if not inception_path.exists():
            raise FileNotFoundError(f"INCEPTION_SMILES_FILE does not exist: {inception_path}")


def prepare_smiles():
    log("Reading SMILES dataset...")
    raw_lines = TL_SMILES_FILE.read_text(encoding="utf-8").splitlines()

    # Preserve the first whitespace-delimited field, remove blanks and duplicates
    # while retaining deterministic ordering before the seeded shuffle.
    seen = set()
    all_smiles = []
    for line in raw_lines:
        if not line.strip():
            continue
        smi = line.split()[0].strip()
        if smi and smi not in seen:
            seen.add(smi)
            all_smiles.append(smi)

    if len(all_smiles) < 2:
        raise ValueError("Need at least 2 unique SMILES to create train/validation splits")

    rng = random.Random(SEED)
    rng.shuffle(all_smiles)

    n_train = int(len(all_smiles) * 0.8)
    n_train = min(max(n_train, 1), len(all_smiles) - 1)
    tl_train = all_smiles[:n_train]
    tl_val = all_smiles[n_train:]

    TL_TRAIN = OUT_DIR / "tl_train.smi"
    TL_VAL = OUT_DIR / "tl_val.smi"
    TL_TRAIN.write_text("\n".join(tl_train) + "\n", encoding="utf-8")
    TL_VAL.write_text("\n".join(tl_val) + "\n", encoding="utf-8")
    log(f"Loaded {len(all_smiles)} unique non-empty SMILES from {TL_SMILES_FILE.name}")
    log(f"Seed={SEED} | 80/20 split done — Train: {len(tl_train)} | Val: {len(tl_val)}")
    log(f"Train set -> {TL_TRAIN}")
    log(f"Val set   -> {TL_VAL}")
    return TL_TRAIN, TL_VAL

def stage1_toml(prior, out):
    """
    Generates the TOML configuration for Stage 1 Reinforcement Learning.
    Focuses on scoring physicochemical properties (QED, TPSA) and penalizing specific structural alerts.
    """
    return f"""\
run_type = "staged_learning"
device   = "{DEVICE}"
tb_logdir       = "{out}/tb_stage1"
json_out_config = "{out}/_stage1.json"

[parameters]
prior_file         = "{prior}"
agent_file         = "{prior}"
summary_csv_prefix = "{out}/stage1"
batch_size         = {BATCH_SIZE}
use_checkpoint     = false

[learning_strategy]
type  = "dap"
sigma = 128
rate  = 0.0001

[[stage]]
max_score  = 1.0
max_steps  = {S1_STEPS}
chkpt_file = "{out}/stage1.chkpt"

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Alerts"
params.smarts = [
    # oversized rings
    "[*;r8]","[*;r9]","[*;r10]","[*;r11]","[*;r12]",
    "[*;r13]","[*;r14]","[*;r15]","[*;r16]","[*;r17]",
    # reactive / unstable
    "[#8][#8]","[#6;+]","[#16][#16]","C#C","C(=[O,S])[O,S]",
    # AMES alerts
    "[#7;!n][#7;!n]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[NX3;H2;$(Nc1ccccc1)]",
    "[N+](=O)[O-]",
    "[#7;!n]=[#7;!n]",
    "[N+](=O)[N+](=O)[N-]"
]

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name   = "QED"
weight = 0.6

[[stage.scoring.component]]
[stage.scoring.component.TPSA]
[[stage.scoring.component.TPSA.endpoint]]
name   = "TPSA"
weight = 0.4
transform.type = "reverse_sigmoid"
transform.high = 90.0
transform.low  = 20.0
transform.k    = 0.5
"""

def tl_toml(input_model, output_model, train_smiles, val_smiles, out):
    """
    Generates the TOML configuration for the Transfer Learning phase.
    Uses the specified dataset to adapt the model to the target chemical space.
    """
    return f"""\
run_type  = "transfer_learning"
device    = "{DEVICE}"
tb_logdir = "{out}/tb_tl"

[parameters]
num_epochs             = {TL_EPOCHS}
save_every_n_epochs    = 1
batch_size             = {BATCH_SIZE}
sample_batch_size      = 500

input_model_file       = "{input_model}"
output_model_file      = "{output_model}"
smiles_file            = "{train_smiles}"
validation_smiles_file = "{val_smiles}"

standardize_smiles   = true
randomize_smiles     = true
randomize_all_smiles = false
internal_diversity   = true
"""

def stage2_toml(prior, agent, scorer, chemprop_py, out):
    """
    Generates the TOML configuration for Stage 2 Reinforcement Learning.
    Incorporates an external chemprop model(MPNN) to guide the generation
    towards high-affinity 5-HT2A compounds, while biasing towards ADMET-friendly
    physicochemical profiles (low logP, low HBD, few aromatic rings).
    """
    inception_file = resolve_optional_path(INCEPTION_SMILES_FILE)
    return f"""\
run_type = "staged_learning"
device   = "{DEVICE}"
tb_logdir       = "{out}/tb_stage2"
json_out_config = "{out}/_stage2.json"

[parameters]
prior_file         = "{prior}"
agent_file         = "{agent}"
summary_csv_prefix = "{out}/stage2"
batch_size         = {BATCH_SIZE}
use_checkpoint     = {str(USE_CHECKPOINT).lower()}

[learning_strategy]
type  = "dap"
sigma = 128
rate  = 0.0001

[diversity_filter]
type        = "IdenticalMurckoScaffold"
bucket_size = 10
minscore    = 0.6

[inception]
smiles_file = "{inception_file}"
memory_size = 50
sample_size = 10

[[stage]]
max_score  = 1.0
max_steps  = {S2_STEPS}
chkpt_file = "{out}/stage2.chkpt"

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name   = "ChemProp_5HT2A"
weight = 0.35
params.executable = "{chemprop_py}"
params.args       = "{scorer}"
params.property = "score"

# Eficacia (XGBoost prob_AGONIST) — DESACTIVADO: el componente de eficacia clasifica
# mal compuestos clave del panel (psilocina 0.104, 6-F-DET 0.994); ver plan ADR-007.
# Se reactiva cuando el clasificador se re-entrene con datos de Emax curados.
# [[stage.scoring.component]]
# [stage.scoring.component.ExternalProcess]
# [[stage.scoring.component.ExternalProcess.endpoint]]
# name   = "Efficacy_5HT2A"
# weight = 0.35
# params.executable = "{chemprop_py}"
# params.args       = "{scorer}"
# params.property = "prob_AGONIST"

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Alerts"
params.smarts = [
    # oversized rings
    "[*;r8]","[*;r9]","[*;r10]","[*;r11]","[*;r12]",
    "[*;r13]","[*;r14]","[*;r15]","[*;r16]","[*;r17]",
    # reactive / unstable
    "[#8][#8]","[#6;+]","[#16][#16]","C#C","C(=[O,S])[O,S]",
    # AMES alerts
    "[#7;!n][#7;!n]",                     # hydrazine
    "[#7;!n][S;!$(S(=O)=O)]",             # amine-sulfur
    "[NX3;H2;$(Nc1ccccc1)]",              # aniline (Ar-NH2)
    "[N+](=O)[O-]",                        # nitro group
    "[#7;!n]=[#7;!n]",                    # azo
    "[N+](=O)[N+](=O)[N-]"               # nitrate ester
]

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name   = "QED"
weight = 0.20

[[stage.scoring.component]]
[stage.scoring.component.TPSA]
[[stage.scoring.component.TPSA.endpoint]]
name   = "TPSA"
weight = 0.10
transform.type = "reverse_sigmoid"
transform.high = 90.0
transform.low  = 20.0
transform.k    = 0.5

[[stage.scoring.component]]
[stage.scoring.component.SlogP]
[[stage.scoring.component.SlogP.endpoint]]
name   = "SlogP"
weight = 0.15
transform.type = "double_sigmoid"
transform.low  = 1.0
transform.high = 4.0
transform.coef_div = 100.0
transform.coef_si  = 150.0
transform.coef_se  = 150.0

[[stage.scoring.component]]
[stage.scoring.component.HBondDonors]
[[stage.scoring.component.HBondDonors.endpoint]]
name   = "HBondDonors"
weight = 0.10
transform.type = "left_step"
transform.low  = 3.0
"""



def validate_stage2_scoring_config(config_text: str):
    """Basic configuration sanity checks; does not modify reward semantics."""
    required_tokens = [
        '[stage.scoring]',
        'type = "geometric_mean"',
        'name   = "ChemProp_5HT2A"',
    ]
    missing = [token for token in required_tokens if token not in config_text]
    if missing:
        raise ValueError(f"Stage 2 scoring config is missing expected tokens: {missing}")

def select_best_tl_checkpoint(log_file, checkpoints):
    """
    Selects the TL checkpoint with the lowest validation NLL.
    Parses the REINVENT4 TL log to find per-epoch validation metrics.
    Fails loudly when the log cannot be parsed or the matching checkpoint is absent.
    """
    import re

    log_path = Path(log_file)
    if not log_path.exists():
        raise FileNotFoundError(f"TL log not found: {log_path}")

    log_text = log_path.read_text()
    epoch_nll = {}

    # REINVENT4 TL logs: "  epoch  total | train_nll | valid_nll | ..."
    for match in re.finditer(
        r'^\s*(\d+)\s+\d+\s*\|\s*[\d.]+\s*\|\s*([\d.]+)', log_text, re.MULTILINE
    ):
        epoch = int(match.group(1))
        val_nll = float(match.group(2))
        epoch_nll[epoch] = val_nll

    # REINVENT4 4.7.x also reports only the selected minimum in plain text.
    if not epoch_nll:
        match = re.search(
            r"Best validation loss \(([\d.]+)\) was at epoch (\d+)", log_text
        )
        if match:
            epoch_nll[int(match.group(2))] = float(match.group(1))

    if not epoch_nll:
        raise RuntimeError(
            f"Could not parse validation NLL from TL log: {log_path}. "
            "Refusing to silently select the last checkpoint."
        )

    best_epoch = min(epoch_nll, key=epoch_nll.get)
    log(f"  Best validation NLL = {epoch_nll[best_epoch]:.2f} at epoch {best_epoch}")

    # Match checkpoint to best epoch
    for ckpt in checkpoints:
        if f".{best_epoch}.chkpt" in ckpt.name:
            return ckpt

    raise FileNotFoundError(
        f"Validation selected epoch {best_epoch}, but no matching checkpoint was found in {log_path.parent}."
    )

def get_start_step():
    configured = os.getenv("START_STEP", "").strip()
    if configured:
        try:
            step = int(configured)
        except ValueError as exc:
            raise ValueError("START_STEP must be an integer between 1 and 4") from exc
        if 1 <= step <= 4:
            return step
        raise ValueError("START_STEP must be an integer between 1 and 4")

    print()
    print("Available starting steps:")
    print("  1: Data Preparation (SMILES split)")
    print("  2: Stage 1 RL (Physicochemical optimization)")
    print("  3: Transfer Learning (Domain adaptation)")
    print("  4: Stage 2 RL (Target-guided optimization with Chemprop)")
    while True:
        try:
            step = int(input("Start from step (1-4): "))
            if 1 <= step <= 4:
                return step
        except ValueError:
            pass
        print("Please enter a number between 1 and 4.")

def get_stop_step(start_step):
    """Optional STOP_STEP (1-4) to run only up to a given stage.

    Defaults to 4 (full pipeline). Used by the Snakemake workflow to split
    Stage 1 (STOP_STEP=1) from TL + Stage 2 (START_STEP=2).
    """
    configured = os.getenv("STOP_STEP", "").strip()
    if not configured:
        return 4
    try:
        step = int(configured)
    except ValueError as exc:
        raise ValueError("STOP_STEP must be an integer between 1 and 4") from exc
    if not 1 <= step <= 4:
        raise ValueError("STOP_STEP must be an integer between 1 and 4")
    if step < start_step:
        raise ValueError(f"STOP_STEP ({step}) must be >= START_STEP ({start_step})")
    return step

def main():
    banner("Staged Learning Pipeline — 5-HT2A De Novo Generation")
    validate_inputs()

    log(f"Seed             : {SEED}")
    log(f"Output directory : {OUT_DIR}")
    log(f"Device           : {DEVICE}")
    log(f"Batch size       : {BATCH_SIZE}")
    log(f"Prior model      : {PRIOR_FILE}")
    log(f"SMILES dataset   : {TL_SMILES_FILE}")
    log(f"Scorer script    : {SCORER_SCRIPT}")
    NUM_WORKERS = os.cpu_count() or 1
    log(f"Threads          : {NUM_WORKERS}")

    START_STEP = get_start_step()
    STOP_STEP = get_stop_step(START_STEP)
    log(f"Starting pipeline from step {START_STEP}/4 to step {STOP_STEP}/4")

    # ── Step 1: Data preparation ──────────────────────────────────────────
    if START_STEP <= 1 <= STOP_STEP:
        banner("Step 1/4: Data Preparation")
        log("Reading source SMILES, cleaning, and performing 80/20 train/val split")
        log(f"Source: {TL_SMILES_FILE}")
        tl_train, tl_val = prepare_smiles()
        log("Data preparation complete.")
    else:
        tl_train = OUT_DIR / "tl_train.smi"
        tl_val   = OUT_DIR / "tl_val.smi"
        if not tl_train.exists() or not tl_val.exists():
            log(f"ERROR: Cannot skip step 1 — {tl_train.name} or {tl_val.name} not found")
            sys.exit(1)
        log(f"Using existing data: {tl_train.name}, {tl_val.name}")

    # ── Step 2: Stage 1 RL ────────────────────────────────────────────────
    if START_STEP <= 2 <= STOP_STEP:
        banner("Step 2/4: Stage 1 — Reinforcement Learning (Physicochemical)")
        log("Goal: Pre-train the agent on general molecular properties")
        log("      (QED, TPSA) while penalizing unwanted substructures.")
        log(f"Configuration: {S1_STEPS} steps, batch_size={BATCH_SIZE}")
        shutil.rmtree(OUT_DIR / "tb_stage1", ignore_errors=True)
        s1_cfg = OUT_DIR / "stage1.toml"
        s1_cfg.write_text(stage1_toml(PRIOR_FILE, OUT_DIR))
        run_step("Stage 1 RL",
                 ["reinvent", "-l", str(OUT_DIR / "stage1.log"), str(s1_cfg)],
                 check_file=OUT_DIR / "stage1.chkpt",
                 desc="Reinforcement learning optimizing QED + TPSA")
    else:
        s1_ckpt = OUT_DIR / "stage1.chkpt"
        if not s1_ckpt.exists():
            log(f"ERROR: Cannot skip step 2 — {s1_ckpt.name} not found")
            sys.exit(1)
        log(f"Using existing checkpoint: {s1_ckpt.name}")

    # ── Step 3: Transfer Learning ─────────────────────────────────────────
    if START_STEP <= 3 <= STOP_STEP:
        banner("Step 3/4: Transfer Learning (Domain Adaptation)")
        log("Goal: Fine-tune the agent on the target SMILES dataset to")
        log("      focus generation towards the 5-HT2A chemical space.")
        log(f"Configuration: {TL_EPOCHS} epochs, batch_size={BATCH_SIZE}")
        shutil.rmtree(OUT_DIR / "tb_tl", ignore_errors=True)
        tl_output = OUT_DIR / "tl.model"
        tl_cfg = OUT_DIR / "tl.toml"
        tl_cfg.write_text(tl_toml(OUT_DIR / "stage1.chkpt", tl_output, tl_train, tl_val, OUT_DIR))
        run_step("Transfer Learning",
                 ["reinvent", "-l", str(OUT_DIR / "tl.log"), str(tl_cfg)],
                 desc="Fine-tuning on target SMILES via transfer learning")
        tl_ckpts = sorted(OUT_DIR.glob("tl.model.*.chkpt"))
        if not tl_ckpts:
            log("ERROR: No checkpoints generated for Transfer Learning")
            sys.exit(1)
        tl_model = select_best_tl_checkpoint(OUT_DIR / "tl.log", tl_ckpts)
        log(f"Selected checkpoint: {tl_model.name}")
    else:
        tl_ckpts = sorted(OUT_DIR.glob("tl.model.*.chkpt"))
        if not tl_ckpts:
            log("ERROR: Cannot skip step 3 — no tl.model.*.chkpt files found")
            sys.exit(1)
        tl_model = select_best_tl_checkpoint(OUT_DIR / "tl.log", tl_ckpts)
        log(f"Using existing checkpoint: {tl_model.name}")

    # ── Step 4: Stage 2 RL ────────────────────────────────────────────────
    if START_STEP <= 4 <= STOP_STEP:
        banner("Step 4/4: Stage 2 — Reinforcement Learning (Target-Guided)")
        log("Goal: Optimize the fine-tuned agent using the Chemprop MPNN")
        log("      scoring function to maximize predicted 5-HT2A affinity,")
        log("      while maintaining drug-likeness (QED, TPSA).")
        log(f"Configuration: {S2_STEPS} steps, batch_size={BATCH_SIZE}")
        if not USE_CHECKPOINT:
            shutil.rmtree(OUT_DIR / "tb_stage2", ignore_errors=True)
        s2_cfg = OUT_DIR / "stage2.toml"
        stage2_text = stage2_toml(
            PRIOR_FILE, tl_model,
            SCORER_SCRIPT.resolve(), CHEMPROP_PYTHON_PATH, OUT_DIR
        )
        validate_stage2_scoring_config(stage2_text)
        s2_cfg.write_text(stage2_text)
        if INCEPTION_SMILES_FILE:
            log(f"Inception initial memory: {INCEPTION_SMILES_FILE}")
        else:
            log("Inception initial memory: empty (INCEPTION_SMILES_FILE not set)")
        run_step("Stage 2 RL",
                 ["reinvent", "-l", str(OUT_DIR / "stage2.log"), str(s2_cfg)],
                 check_file=OUT_DIR / "stage2.chkpt",
                 desc="RL with Chemprop MPNN scoring for 5-HT2A affinity")

    # ── Summary ───────────────────────────────────────────────────────────
    banner("Pipeline Complete")
    csv_files = list(OUT_DIR.glob("stage2_*.csv"))
    if csv_files:
        log(f"Generated {len(csv_files)} CSV files with results:")
        for f in csv_files:
            log(f"  - {f.name}")
    else:
        log("WARNING: No CSV files were generated in Stage 2.")
    log(f"All outputs saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
