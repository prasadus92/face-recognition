#!/usr/bin/env bash
# Reproduces the AT&T/ORL accuracy and latency numbers in docs/benchmarks/orl.md.
# Usage: scripts/benchmark-orl.sh   (ORL_DIR and OUT_DIR override the dataset and report folders)
set -euo pipefail

ORL_URL="https://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"
ORL_SHA256="509d763bd2276aa054d260d7f1000e8be212738134f2f83d8ff13bc2adf1c8ec"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${ORL_DIR:-$ROOT/data/orl}"
OUT="${OUT_DIR:-$ROOT/target/benchmarks/orl}"

JAVA="${JAVA_HOME:+$JAVA_HOME/bin/}java"

sha256() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    echo "Need sha256sum or shasum to verify the download" >&2
    exit 1
  fi
}

fetch_dataset() {
  if [[ -f "$DATA/s40/10.pgm" ]]; then
    return
  fi
  TMP="$(mktemp -d)"
  trap 'rm -rf "$TMP"' EXIT
  local tmp="$TMP"
  echo "Downloading AT&T/ORL (3.6 MB) from AT&T Laboratories Cambridge"
  curl -fsSL -o "$tmp/att_faces.zip" "$ORL_URL"
  if [[ "$(sha256 "$tmp/att_faces.zip")" != "$ORL_SHA256" ]]; then
    echo "Checksum mismatch for att_faces.zip" >&2
    exit 1
  fi
  mkdir -p "$DATA"
  unzip -q -o "$tmp/att_faces.zip" -d "$DATA"
}

benchmark() {
  "$JAVA" -jar "$JAR" benchmark "$DATA" --format orl --components 40 "$@"
}

summary() {
  sed -n "/$1 RESULTS ---/,/^\$/p"
}

fetch_dataset
(cd "$ROOT" && ./mvnw -q -ntp -DskipTests clean package)
JAR="$(ls "$ROOT"/target/face-recognition-*-exec.jar)"

echo "== 10-fold stratified cross-validation (9 train / 1 test per person), Euclidean"
benchmark --type cross_validation --folds 10 -o "$OUT/10-fold" | summary CROSS-VALIDATION
echo "== 2-fold stratified cross-validation (5 train / 5 test per person), Euclidean"
benchmark --type cross_validation --folds 2 -o "$OUT/2-fold" | summary CROSS-VALIDATION
echo "== LBPH with chi-square distance, 10-fold and 2-fold"
benchmark --type cross_validation --folds 10 -a lbph -m chi_square -o "$OUT/10-fold-chi2" | summary CROSS-VALIDATION
benchmark --type cross_validation --folds 2 -a lbph -m chi_square -o "$OUT/2-fold-chi2" | summary CROSS-VALIDATION
echo "== Feature extraction latency"
benchmark --type performance -o "$OUT/latency" | summary PERFORMANCE
echo "Full reports: $OUT"
