"""Smoke test sin dependencias pesadas: valida grafo y configs del monorepo."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def test_snakefile_includes_all_rules():
    text = (ROOT / "workflow" / "Snakefile").read_text()
    for r in ["rules/chemprop.smk", "rules/generacion.smk", "rules/filter.smk", "rules/admet.smk"]:
        assert r in text, f"falta include {r}"

def test_no_docking_rules():
    blob = "".join(p.read_text() for p in (ROOT / "workflow" / "rules").glob("*.smk"))
    blob += (ROOT / "workflow" / "Snakefile").read_text()
    low = blob.lower()
    for banned in ["gnina", "autobox", "redock", "cnn_scoring", "pharmit candidates"]:
        assert banned not in low, f"docking debe quedar afuera (hallado {banned})"

def test_envs_pinned():
    for env in (ROOT / "workflow" / "envs").glob("*.yaml"):
        text = env.read_text()
        assert "==" in text, f"{env.name} debe pinear versiones con =="

def test_gitignore_bloquea_ckpt():
    gi = (ROOT / ".gitignore").read_text()
    assert "*.ckpt" in gi and "results/" in gi

if __name__ == "__main__":
    test_snakefile_includes_all_rules()
    test_no_docking_rules()
    test_envs_pinned()
    test_gitignore_bloquea_ckpt()
    print("smoke OK")
