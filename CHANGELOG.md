# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- `POST /api/v1/recognize` honors its `threshold` and `maxAlternatives` parameters. Both were documented and ignored. Without them the configured `recognition.threshold` and `recognition.max-alternatives` apply; the latter was read by nothing before. A threshold outside 0 to 1 (or NaN) or a negative limit returns 400. The classifier reports at most `classification.k-neighbors` minus 1 runners-up for a match.
- A malformed LFW `pairs.txt` line raises an `IOException` naming the line instead of a bare `NumberFormatException`.
- Client-supplied names, ids, methods and URIs are stripped of line breaks before they are logged (log injection).
- Saved and imported models now recognize faces. A model file used to hold only each identity's average feature vector, not the trained extractor, so after a restart or `POST /api/v1/model/import` the service reported itself trained and then failed on the first recognition. Model files now carry the trained extractor, every identity and the preprocessed face crops, so a restored service also keeps its identities when it is retrained.
- The command-line tool works. The executable jar's main class only started Spring, so no CLI command was reachable; `train` failed with `NotSerializableException` because it serialized the whole service; none of the commands could read PGM files. `java -jar face-recognition-exec.jar <command>` now runs the CLI and the jar without a command starts the REST API.
- `POST /api/v1/model/import` no longer deserializes arbitrary classes from the upload. Decoding runs behind an `ObjectInputFilter` allowlist of the exact classes a model contains, with depth, reference, array and stream-size limits. An import that fails leaves the current model in place and returns 400.

### Changed
- Reproducible AT&T/ORL benchmark: `scripts/benchmark-orl.sh` downloads and checksums the dataset and runs 10-fold and 2-fold cross-validation plus a latency run. Results are in `docs/benchmarks/orl.md`.
- `benchmark --metric` picks the nearest-neighbor distance (`EUCLIDEAN`, `COSINE`, `MANHATTAN`, `CHI_SQUARE`).
- `FaceRecognitionService#recognize(FaceImage, double)` takes a per-call threshold.
- `POST /api/v1/enroll` drops `sourceDescription`, which was never stored.
- `FaceDetector` drops `supportsLandmarks()` and `detectLandmarks(...)`; no caller used them and both detectors returned nothing. `RecognitionResult.MatchResult` is no longer `Comparable`; `getTopAlternatives` sorts by confidence explicitly.
- **Relicensed from GNU General Public License v3.0 to Apache License, Version 2.0.** The project was originally released under GPL-3.0 in 2014. The relicensing was legally permissible because all copyrightable contributions up to this point were made by the sole copyright holder, so no third-party consent was required. Apache 2.0 matches the ecosystem default for Java libraries (Spring Boot, Jackson, Micrometer, Bucket4j, picocli, springdoc are all Apache 2.0) and removes the copyleft adoption friction that came with GPL-3.0. See `License.txt` for the full Apache 2.0 text and the transition note.
- **Spring Boot 3.2.5 → 4.0.5** (and springdoc 2.3.0 → 3.0.3). See the dedicated 4.0 upgrade commit for the detailed API-change list.
- Java 25 LTS baseline (was 17), Spring Boot 4.1.2, springdoc 3.2.0, Bucket4j 8.22.0 (`bucket4j_jdk17-core`), Checkstyle 14.4.0, SpotBugs 4.10.5, JaCoCo 0.8.16 and current Maven plugins.
- Maven Wrapper (Maven 3.10.1); CI, CodeQL and the Docker build use `./mvnw`.
- Docker image on `eclipse-temurin:25-jre-alpine` with Spring Boot layered extraction. The healthcheck probes liveness, so a container with no trained model is still healthy. JVM flags now come from `JDK_JAVA_OPTIONS`; the old `JAVA_OPTS` variable is no longer read.
- CI builds and smoke-tests the Docker image on every pull request; the image job previously targeted the nonexistent `master` branch and never ran. The release job now triggers on `v*` tags, which the old workflow never listened to.
- Model file format version 3. Files written by earlier versions are rejected with a message to retrain; they could not recognize faces anyway.
- The persistence package is now `TrainedModel` (a record), `ModelCodec`, `ModelRepository` (save and load by name) and `FileModelRepository` (atomic writes). The unused backup, restore, listing and caching APIs are gone.
- `FaceRecognitionService#snapshot()` and `#loadModel(TrainedModel)` are replaced by `exportModel()` and `importModel(byte[])`.
- Enrollment detects, aligns and crops the face when the image is enrolled, not at training time, so a failing detector now surfaces from `enroll`.
- A restored model keeps each identity's id, external id, active flag and metadata. Restores used to create new ids and reactivate soft-deleted identities.
- `POST /api/v1/model/import` drops the `merge` parameter, which never merged.
- The CLI is `train`, `enroll`, `recognize` and `benchmark`. Model files are the same format the REST API exports. `serve` (a second, hand-written HTTP server) and the `export`/`import` commands are gone; run the jar without a command for the API.
- `FaceImage.fromFile` reads PGM (P5 and P2), shared by the CLI, the service and the benchmark loader.
- The CLI and the Spring auto-configuration build the pipeline through one `FaceRecognitionServiceFactory`.
- `BenchmarkRunner` lost its hand-parsed `main`; the `benchmark` command is its front end. The `benchmarks` Maven profile is gone. Defaults changed: threshold 0 (closed-set rank-1 accuracy, was 0.6), 10 folds (was 5), reports in `benchmark-results/` (was `./benchmark_results`). `--warmup`, `--runs` and `--verbose` are gone.
- `recognize` reports an unreadable image as an error line and carries on, exiting 1. For an unmatched image it prints the nearest person's confidence instead of 0.
- Model files written by the CLI and by `FileModelRepository` are replaced atomically and keep the permissions of the file they replace (`rw-r--r--` when new).
- PGM images larger than 4096 pixels per side, or with samples above their declared maximum, are rejected before allocation.
- The CLI logs warnings to stderr, so `recognize --json` output stays parseable. `--version` reads the jar manifest.
- The log line for a service built without a face detector drops from WARN to INFO.
- Benchmark reports use Jackson 3. The `jackson-datatype-jsr310` and `picocli-spring-boot-starter` dependencies are gone.

## [2.1.0]

### Added
- **Java 17 / Spring Boot 3.2** baseline. `javax.*` → `jakarta.*` across validation and servlet APIs.
- **Central `FaceRecognitionProperties`** bound from `application.yml` via `@ConfigurationProperties`, plus a `FaceRecognitionAutoConfiguration` that wires the `FaceDetector`, `FeatureExtractor`, `FaceClassifier`, `ModelRepository`, and `FaceRecognitionService` beans from configuration. No more hardcoded defaults drifting from the YAML.
- **Experimental ONNX deep-learning backend** — `OnnxDeepFeatureExtractor` scaffold implementing `FeatureExtractor` for FaceNet/ArcFace-style embeddings. Bring-your-own-weights.
- **Request correlation** — `RequestIdFilter` stamps `X-Request-ID` / MDC `traceId` on every request and every log line.
- **Rate limiting** — per-IP Bucket4j token bucket filter, configurable via `facerecognition.ratelimit.*`.
- **Custom Micrometer metrics** — `facerecognition.detect`, `facerecognition.extract`, `facerecognition.match`, `facerecognition.recognize.total`, counters for `enrollments`, `recognitions`, `errors`.
- **`ModelReadyHealthIndicator`** reporting `UP` only when `FaceRecognitionService#isTrained()` is true.
- **Model persistence wiring** — auto-load at startup / auto-save after training, driven by `facerecognition.model.*`.
- **Thread-safe `FaceRecognitionService`** — enrolment and training now guarded by a `ReadWriteLock`; detector is required (fail-loud instead of silently bypassing detection).
- **REST integration tests** (MockMvc) for enrolment, recognition, training, and identity CRUD endpoints, including error-path coverage.
- **Quality gates** — enforcing Checkstyle + SpotBugs + JaCoCo coverage floor in CI (no more `continue-on-error`). Added CodeQL, Dependabot, CycloneDX SBOM generation, and a release workflow.
- **Community files** — `SECURITY.md`, `CODE_OF_CONDUCT.md`, `SUPPORT.md`, `CODEOWNERS`, `.editorconfig`, GitHub issue forms.
- Truthful `README`, `ROADMAP`, and benchmark documentation.

### Changed
- **REST API** responses now always flow through `GlobalExceptionHandler` (no inline `IllegalArgumentException` in controllers). Validation annotations moved onto DTOs. `/identities` endpoint is paginated.
- **`FaceRegion#equals/hashCode`** now include `confidence` to preserve `Set`/`Map` semantics.
- **`RecognitionResult#getDistance`** now returns `Optional<Double>` instead of a `Double.MAX_VALUE` sentinel.
- **Eigenfaces / Fisherfaces** — deterministic eigenvector ordering and improved numerical stability around the `numSamples < numPixels` projection path.
- **KNN confidence** — replaced magic per-metric scale factors with a calibrated normalizer whose parameters are derived at enrolment time.

### Removed
- Dead `platform-specific/windows/*.dll` artefacts and the `scripts/setup-libs.sh` Java 3D / JAI downloader.
- Legacy `TSCD` / `FrontEnd` references from docs.
- Duplicate `DatasetLoader` in `benchmark/` (benchmark now depends on the `infrastructure.dataset` one).
- Incorrect "90% test coverage" / "99% LFW accuracy" / "Maven Central install" claims from the README.

### Security
- Bumped all transitive dependencies via Spring Boot 3.2.5 and Jackson 2.17.
- Added `SECURITY.md` with a private disclosure channel.
- CodeQL + secret-scanning workflows enabled.

## [2.0.0] - 2024-01

### Added
- Initial clean-architecture restructure under `com.facerecognition` (`domain` / `application` / `infrastructure` / `api`).
- Domain model: `FaceImage`, `FaceRegion`, `FaceLandmarks`, `FeatureVector`, `Identity`, `RecognitionResult`.
- `Fisherfaces` and `LBPH` feature extractors.
- `KNNClassifier` with Euclidean / Cosine / Manhattan / Chi-square metrics.
- `FaceRecognitionService` orchestrator with builder API.
- Spring Boot REST API + Swagger UI.
- Picocli CLI with `enroll`, `train`, `recognize`, `benchmark`, `serve` commands.
- Multi-stage `Dockerfile`.
- Quality metrics (brightness, contrast, sharpness) on `FaceImage`.

### Changed
- Rewrote the Eigenfaces implementation with proper Javadoc and the high-dim PCA trick.

## [1.0.0] - 2014

### Added
- Initial release: Eigenfaces (PCA) + Two-Stage Classification and Detection (TSCD), Swing GUI, skin-colour detection, k-NN, MySQL user auth.

---

## Links

- [GitHub Repository](https://github.com/prasadus92/face-recognition)
- [Issue Tracker](https://github.com/prasadus92/face-recognition/issues)
- [Discussions](https://github.com/prasadus92/face-recognition/discussions)
