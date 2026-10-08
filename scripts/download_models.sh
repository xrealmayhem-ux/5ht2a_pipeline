#!/usr/bin/env bash
# Descarga modelos desde Zenodo a modelos/ (reemplaza el symlink cross-repo).
# Uso: bash scripts/download_models.sh --dest modelos
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

if [[ -z "$RECORD" || "$RECORD" == "REEMPLAZAR_POR_RECORD_ZENODO" ]]; then
  echo "ERROR: define el record Zenodo en config/config.yaml (zenodo.record)" >&2
  echo "o pásalo con --record <id>. Ver chemeleon/README (foundation 15460715)." >&2
  exit 2
fi

mkdir -p "$DEST/chemeleon"
echo "Descargando modelos Zenodo record=$RECORD -> $DEST"

# Ejemplo con zenodo_get (ajustar nombres reales al publicar):
# zenodo_get -r "$RECORD" -o "$DEST"
# Alternativa curl:
# curl -L "https://zenodo.org/api/records/${RECORD}/files-archive" -o "${DEST}/models.zip"

echo "TODO: completar URLs al publicar el record. Verificá con:"
echo "  sha256sum -c scripts/checksums.txt"
