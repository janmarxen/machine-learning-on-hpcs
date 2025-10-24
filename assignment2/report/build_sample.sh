#!/usr/bin/env bash
# Lightweight script to compile the sample LaTeX report only.
# It avoids running the larger Makefile that builds templates and class files.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLES_DIR="$ROOT_DIR/samples"
OUT_DIR="$ROOT_DIR/output"
# Allow overriding the tex file: ./build_sample.sh sample-sigplan.tex
TEXFILE="${1:-sample-sigplan.tex}"

if [ ! -x "$(command -v pdflatex 2>/dev/null || true)" ] && [ -z "$(command -v pdflatex 2>/dev/null || true)" ]; then
  echo "pdflatex not found in PATH. Please install TeX Live (pdflatex) and retry."
  exit 2
fi

mkdir -p "$OUT_DIR"

echo "Compiling $TEXFILE -> $OUT_DIR"

pushd "$SAMPLES_DIR" >/dev/null

# Run pdflatex twice to resolve references; stop on error and copy PDF to OUT_DIR
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT_DIR" "$TEXFILE"
pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT_DIR" "$TEXFILE" || true

PDF_PATH="$OUT_DIR/${TEXFILE%.tex}.pdf"

if [ -f "$PDF_PATH" ]; then
  echo "PDF generated: $PDF_PATH"
  ls -lh "$PDF_PATH"
else
  echo "PDF not generated. Check pdflatex output above for errors." >&2
  exit 3
fi

popd >/dev/null

echo "Done. You can open the file with your preferred PDF viewer or serve it from the project." 
