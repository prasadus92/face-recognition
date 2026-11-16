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
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

fetch_dataset() {
  if [[ -f "$DATA/s40/10.pgm" ]]; then
    return
  fi
  if ! command -v sha256sum >/dev/null && ! command -v shasum >/dev/null; then
    echo "Need sha256sum or shasum to verify the download" >&2
    exit 1
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

# Gallery: photos 1-5 of people 1-20. Genuine probes: their photos 6-10. Strangers: all photos of people 21-40.
prepare_open_set() {
  local root="$OUT/open-set"
  rm -rf "$root"
  mkdir -p "$root/gallery" "$root/genuine" "$root/strangers"
  for person in $(seq 1 40); do
    for photo in $(seq 1 10); do
      local source="$DATA/s$person/$photo.pgm"
      if (( person > 20 )); then
        ln -s "$source" "$root/strangers/s$person-$photo.pgm"
      elif (( photo > 5 )); then
        ln -s "$source" "$root/genuine/s$person-$photo.pgm"
      else
        mkdir -p "$root/gallery/s$person"
        ln -s "$source" "$root/gallery/s$person/$photo.pgm"
      fi
    done
  done
}

# Prints how many genuine probes are matched to the right person and how many strangers are accepted.
open_set() {
  local root="$OUT/open-set" model="$OUT/open-set/$1.frm"
  "$JAVA" -jar "$JAR" train "$root/gallery" -o "$model" -a "$1" -c 40 >/dev/null
  local genuine strangers
  genuine="$("$JAVA" -jar "$JAR" recognize -m "$model" "$root/genuine" | awk -F'\t' '
    { n = split($1, p, "/"); who = p[n]; sub(/-.*/, "", who); if ($2 == who) ok++ }
    END { printf "%d of %d", ok, NR }')"
  strangers="$("$JAVA" -jar "$JAR" recognize -m "$model" "$root/strangers" | awk -F'\t' '
    $2 !~ /^(unknown|no_face_detected|error)/ { accepted++ }
    END { printf "%d of %d", accepted, NR }')"
  printf "%-12s genuine matched %s, strangers accepted %s\n" "$1" "$genuine" "$strangers"
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
echo "== Open set at the default 0.6 threshold (20 enrolled people, 20 strangers)"
prepare_open_set
for algorithm in eigenfaces fisherfaces lbph; do
  open_set "$algorithm"
done
echo "Full reports: $OUT"
