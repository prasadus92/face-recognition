# AT&T/ORL benchmark

The AT&T Database of Faces (formerly the ORL database) has 400 grayscale photos of 40 people, 10 each, at 92x112 pixels. AT&T Laboratories Cambridge took them between 1992 and 1994 with changes in lighting, expression, glasses and head pose. It is the classic first benchmark for Eigenfaces-era methods. The dataset is courtesy of AT&T Laboratories Cambridge.

## Results

Closed-set identification: every test photo belongs to someone in the training set, and the score is the share of test photos whose nearest training photo is the right person (rank-1 accuracy, threshold 0).

| Algorithm | Distance | 10-fold (9 train / 1 test per person) | 2-fold (5 train / 5 test per person) |
|---|---|---|---|
| Eigenfaces | Euclidean | 98.25% (SD 1.95 pp) | 94.00% |
| Fisherfaces | Euclidean | 96.50% (SD 4.21 pp) | 94.00% |
| LBPH | Euclidean | 98.00% (SD 2.45 pp) | 92.25% |
| LBPH | Chi-square | **99.00%** (SD 1.66 pp) | **94.25%** |

SD is the population standard deviation of the 10 fold accuracies, in percentage points. Eigenfaces and Fisherfaces tie on the 2-fold split by coincidence: they predict different labels, with macro precision of 95.2% and 94.1%.

Time to identify one 48x64 face against a gallery of 280 training photos, after a warm-up, on an Apple M1 Pro with OpenJDK 25.0.5 (mean of 12,000 runs):

| Algorithm | Feature extraction | Nearest-neighbor match | Total |
|---|---|---|---|
| Eigenfaces | 0.46 ms | 0.02 ms | 0.48 ms |
| Fisherfaces | 0.45 ms | 0.01 ms | 0.46 ms |
| LBPH | 0.31 ms | 5.79 ms | 6.11 ms |

LBPH extracts fastest, but its 16,384-value histograms make every comparison expensive. Eigenfaces and Fisherfaces compare 40 and 39 numbers.

## Open set: telling strangers apart

Closed-set accuracy assumes everyone is enrolled. In practice a recognizer also has to answer "unknown". Here the gallery holds photos 1 to 5 of people 1 to 20; the genuine probes are their photos 6 to 10, and the strangers are all 200 photos of people 21 to 40. Everything runs through the CLI with 40 components, Euclidean distance for all 3 algorithms (the CLI has no metric option yet) and the default threshold of 0.6.

| Algorithm | Genuine probes matched to the right person | Strangers accepted |
|---|---|---|
| Eigenfaces | 73 of 100 | 10 of 200 |
| Fisherfaces | 68 of 100 | 0 of 200 |
| LBPH | 58 of 100 | 0 of 200 |

Confidence is calibrated on the gallery. For each enrolled photo the classifier measures the distance to the nearest other photo of the same person and to the nearest photo of anyone else. A logistic curve centered halfway between the 2 medians gives about 0.98 at the first and 0.02 at the second.

Before this calibration the score used a fixed scale that put every face above 0.6, so all 200 strangers were accepted by all 3 algorithms. Both scores fall as distance grows, so the ranking, and the trade-off available across thresholds, is unchanged. What changes is that the default threshold lands at a sensible distance for whatever gallery is enrolled. Raise it to reject more strangers, lower it to accept more genuine probes.

The calibration reads distances from the gallery photos the extractor was trained on. Fisherfaces pulls those photos together by design. On this split the median distance from a gallery photo to the nearest other gallery photo of the same person is 0.07, against 0.46 from a new photo of that person. The curve therefore sits too tight for Fisherfaces, which is why it rejects more genuine probes than the others. Eigenfaces (0.44 against 0.52) and LBPH (1.63 against 1.76) are less affected.

## Setup## Setup

- Photos are resized to 48x64 pixels and converted to grayscale.
- Eigenfaces keeps 40 components. Fisherfaces keeps 39, the most LDA allows for 40 people.
- LBPH uses radius 1, 8 neighbors and an 8x8 grid of 256-bin histograms.
- The classifier is a nearest neighbor over every training photo. Calibration uses at most 500 gallery photos, picked at a fixed stride, which keeps it under a second for Eigenfaces even on large galleries.
- Folds are stratified by person and shuffled with seed 42, so every run gives the same numbers.
- Measured on 2026-11-16 with `scripts/benchmark-orl.sh` as of pull request #50.

## Reproduce

```bash
scripts/benchmark-orl.sh
```

The script downloads the dataset from AT&T Laboratories Cambridge, checks its SHA-256, builds the jar and runs the 5 benchmarks and the open-set check above. It takes about 4 minutes on an M1 Pro. HTML, JSON, Markdown and CSV reports land in `target/benchmarks/orl/`. To run one configuration yourself:

```bash
java -jar target/face-recognition-*-exec.jar benchmark data/orl --format orl \
    --type cross_validation --folds 10 --components 40 -a lbph -m chi_square
```

## Reading the numbers

ORL is small, frontal and well lit. The numbers show the implementations work. On photos taken in the wild, such as the LFW dataset, classical methods fall well behind modern face-embedding networks.
