# CLAUDE.md

Guidance for coding agents working in this repository. `AGENTS.md` is a symlink to this file.

## What this is

Face recognition on the JVM: Eigenfaces, Fisherfaces and LBPH, a pure-Java Haar face detector, a Spring Boot REST API and a picocli CLI, all in one executable jar. Java 25, Spring Boot 4.1, Maven.

## Commands

```bash
./mvnw -Pci verify                       # the CI gate: tests, Checkstyle, SpotBugs, JaCoCo floor
./mvnw test -Dtest=KNNClassifierTest     # one test class
./mvnw -DskipTests package               # builds target/face-recognition-<version>-exec.jar
java -jar target/face-recognition-*-exec.jar            # REST API on :8080
java -jar target/face-recognition-*-exec.jar --help     # CLI
scripts/benchmark-orl.sh                 # reproduces docs/benchmarks/orl.md (about 4 minutes)
docker build -t face-recognition .
```

## Layout

- `domain/` models (`FaceImage`, `Identity`, `FeatureVector`, `RecognitionResult`) and the `FaceDetector`, `FeatureExtractor`, `FaceClassifier` interfaces.
- `application/service/FaceRecognitionService` owns enrollment, training, recognition and model export/import under one read-write lock.
- `infrastructure/` implementations: `detection` (Haar cascade, OpenCV parity), `extraction`, `classification` (nearest neighbor with gallery-calibrated confidence), `persistence` (`ModelCodec`, `FileModelRepository`).
- `config/FaceRecognitionServiceFactory` builds the pipeline from `FaceRecognitionProperties`; both the Spring auto-configuration and the CLI use it.
- `api/rest` controllers and DTOs; `api/cli` picocli commands. `FaceRecognitionApplication.main` sends CLI commands to picocli and everything else to Spring.
- `benchmark/` the accuracy, cross-validation and latency harness behind the `benchmark` command.

## Rules that are easy to break

- Model files are Java-serialized behind an allowlist in `ModelCodec`. A new class that ends up inside `TrainedModel` must be added to `ALLOWED_CLASSES`, or every saved model stops loading. Never widen the filter to a package prefix.
- Bump `ModelCodec.FORMAT_VERSION` when the persisted shape changes; old files are rejected with a retrain message.
- The detector matches OpenCV 4.x on the bundled cascade. `HaarCascadeFaceDetectorTest.matchesOpenCv` pins OpenCV's box for `src/test/resources/portrait.png`; if a change moves it, compare against `cv2.CascadeClassifier.detectMultiScale` before updating the expected value.
- Confidence comes from `KNNClassifier`'s gallery calibration, so the REST and CLI thresholds mean "closer than halfway between a typical genuine and a typical impostor distance". Do not reintroduce fixed scales.
- CLI and REST must build the pipeline through `FaceRecognitionServiceFactory` so they cannot drift.
- Numbers in `README.md` and `docs/benchmarks/orl.md` come from `scripts/benchmark-orl.sh`. Rerun it after any change to extraction, classification or preprocessing and update both.

## Conventions

- Every behavior change gets a test. Prove it by removing the change, watching the test fail, and restoring it.
- Comments say what the code cannot: a source, a non-obvious rule, a reason. No comments that restate the code.
- Prefer records and small single-purpose classes. Keep `domain/` free of Spring.
- American English, no em dashes, digits for counts, in code, docs and commit messages.
- Dependencies are pinned to the latest stable release; check Maven Central rather than memory.
- Work through pull requests; `main` only changes by merge, and CI must be green.
