# AT&T/ORL benchmark

The AT&T Database of Faces (formerly the ORL database) has 400 grayscale photos of 40 people, 10 each, at 92x112 pixels. AT&T Laboratories Cambridge took them between 1992 and 1994 with changes in lighting, expression, glasses and head pose. It is the classic first benchmark for Eigenfaces-era methods, and the dataset is courtesy of AT&T Laboratories Cambridge.

## Results

Closed-set identification: every test photo belongs to someone in the training set, and the score is the share of test photos whose nearest training photo is the right person (rank-1 accuracy, threshold 0).

| Algorithm | Distance | 10-fold (9 train / 1 test per person) | 2-fold (5 train / 5 test per person) |
|---|---|---|---|
| Eigenfaces | Euclidean | 98.25% (SD 1.95) | 94.00% |
| Fisherfaces | Euclidean | 96.50% (SD 4.21) | 94.00% |
| LBPH | Euclidean | 98.00% (SD 2.45) | 92.25% |
| LBPH | Chi-square | **99.00%** (SD 1.66) | **94.25%** |

SD is the standard deviation across the 10 folds. Eigenfaces and Fisherfaces tie on the 2-fold split by coincidence; they get different photos wrong (macro precision 95.2% against 94.1%).

Feature extraction for one 48x64 face, after a warm-up, on an Apple M1 Pro with OpenJDK 25.0.5:

| Algorithm | Mean time | Throughput |
|---|---|---|
| Eigenfaces | 0.48 ms | 2,070 faces/s |
| Fisherfaces | 0.46 ms | 2,140 faces/s |
| LBPH | 6.1 ms | 163 faces/s |

## Setup

- Photos are resized to 48x64 pixels and converted to grayscale.
- Eigenfaces keeps 40 components. Fisherfaces keeps 39, the most LDA allows for 40 people.
- LBPH uses radius 1, 8 neighbors and an 8x8 grid of 256-bin histograms.
- The classifier is a nearest neighbor over every training photo.
- Folds are stratified by person and shuffled with seed 42, so every run gives the same numbers.
- Measured on 2026-11-16 at commit `92d8c28`.

## Reproduce

```bash
scripts/benchmark-orl.sh
```

The script downloads the dataset from AT&T Laboratories Cambridge, checks its SHA-256, builds the jar and runs the 5 benchmarks above. It takes about 3 minutes on an M1 Pro. HTML, JSON, Markdown and CSV reports land in `target/benchmarks/orl/`. To run one configuration yourself:

```bash
java -jar target/face-recognition-*-exec.jar benchmark data/orl --format orl \
    --type cross_validation --folds 10 --components 40 -a lbph -m chi_square
```

## Reading the numbers

ORL is small, frontal and well lit. The numbers show the implementations work. On photos taken in the wild, such as the LFW dataset, classical methods fall well behind modern face-embedding networks.
