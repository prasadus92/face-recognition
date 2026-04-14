# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
