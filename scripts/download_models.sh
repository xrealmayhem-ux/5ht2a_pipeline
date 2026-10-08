#!/usr/bin/env bash
# Descarga modelos desde Zenodo a modelos/ (reemplaza el symlink cross-repo).
# Setup productivo single-model (seed_2). Layout creado
# (coincide con scripts/checksums.txt, ~130MB):
#   modelos/chemeleon/checkpoints/pki/seed_2/{pki-best-*.ckpt,scaler.pt}
#   modelos/reinvent_stage2.chkpt
# El umbral p90 de incertidumbre va hardcodeado en scorer_pki.py
# (ENSEMBLE_P90_FALLBACK); no se descargan train_predictions.csv.
# Uso:
#   bash scripts/download_models.sh --dest modelos --record <ZENODO_RECORD_ID>
#   (cd modelos && sha256sum -c ../scripts/checksums.txt)
set -euo pipefail

DEST="modelos"
RECORD=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest) DEST="$2"; shift 2;;
    --record) RECORD="$2"; shift 2;;
    *) echo "arg desconocido: $1" >&2; exit 1;;
  esac
done

if [[ -z "$RECORD" ]]; then
  echo "ERROR: pasa el record Zenodo con --record <id>" >&2
  echo "(foundation CheMeleon externo: record 15460715, solo como referencia)" >&2
  exit 2
fi

API="https://zenodo.org/api/records/${RECORD}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "Consultando $API ..."
FILES="$(curl -fsSL "$API" | python3 -c "
import json,sys
d = json.load(sys.stdin)
for f in d.get('files', d.get('metadata', {}).get('files', [])):
    key = f.get('key') or f.get('filename')
    link = f.get('links', {}).get('self') or f.get('download_url') or f.get('url')
    if key and link:
        print(f'{key}\t{link}')
")"
if [[ -z "$FILES" ]]; then
  echo "ERROR: sin archivos en el record $RECORD" >&2
  exit 3
fi

mkdir -p "$DEST/chemeleon/checkpoints/pki"
while IFS=$'\t' read -r key url; do
  # Mapeo del record 23249654 (nombres planos, Zenodo no acepta "/"):
  #   pki-best-*.ckpt -> chemeleon/checkpoints/pki/seed_2/<mismo nombre>
  #   scaler.pt       -> chemeleon/checkpoints/pki/seed_2/scaler.pt
  #   stage2.chkpt    -> reinvent_stage2.chkpt (canónico: el de Zenodo)
  case "$key" in
    pki-best-*.ckpt)
      target="$DEST/chemeleon/checkpoints/pki/seed_2/$key" ;;
    scaler.pt)
      target="$DEST/chemeleon/checkpoints/pki/seed_2/scaler.pt" ;;
    stage2.chkpt)
      target="$DEST/reinvent_stage2.chkpt" ;;
    *)
      echo "WARN: archivo inesperado en record: $key (se omite)" >&2
      continue ;;
  esac
  mkdir -p "$(dirname "$target")"
  echo " -> $target"
  curl -fSL --retry 3 -o "$target" "$url"
done <<< "$FILES"

echo "Verificando sha256 ..."
(cd "$DEST" && sha256sum -c "$HERE/scripts/checksums.txt")
echo "OK: modelos descargados y verificados en $DEST"
echo "Exporta en tu corrida Snakemake:"
echo "  CHEMPROP_CHEMELEON_DIR=$DEST/chemeleon"
