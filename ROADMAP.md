# Roadmap

A living document describing where the project is today, what is actively being worked on, and what I'd like to land next. Contributions on any item marked `help wanted` are welcome.

## Where we are

| Area | State |
|---|---|
| Clean-architecture layering (`domain` / `application` / `infrastructure` / `api`) | Stable |
| Classical extractors (Eigenfaces, Fisherfaces, LBPH) | Stable |
| KNN classifier + pluggable distance metrics | Stable |
| Spring Boot 3.2 REST API + OpenAPI 3 | Stable |
| Picocli CLI (`enroll`, `train`, `recognize`, `serve`, `benchmark`) | Stable |
| Multi-stage Docker image | Stable |
| Model persistence + auto-save/auto-load | Stable |
| Custom Micrometer metrics + Prometheus endpoint | Stable |
| Request correlation (MDC) + rate limiting | Stable |
| Benchmark harness on bundled micro-dataset | Stable |
| Haar-cascade face detector (real cascade data) | `help wanted` |
| ONNX deep-learning backend (model-agnostic) | Experimental scaffold |
| Publication to Maven Central | Planned |
| JavaDoc hosted on `javadoc.io` | Planned |
| Hosted demo (Render / Fly) | Planned |

## Next milestones

### 2.1 — Enterprise polish *(current branch)*

Goals: make every claim in the README true, close the biggest correctness and observability gaps, tighten CI, and stop advertising vaporware.

- [x] Spring Boot 3.2 / Java 17 baseline; `javax` → `jakarta`
- [x] `@ConfigurationProperties` binding for every runtime knob — `application.yml` becomes the single source of truth
- [x] `FaceRecognitionAutoConfiguration` wires `FaceDetector` / `FeatureExtractor` / `FaceClassifier` / `FaceRecognitionService`
- [x] Thread-safe `FaceRecognitionService` with `ReadWriteLock` and required detector
- [x] Correctness pass: `FaceRegion` equality, `RecognitionResult` distance, Eigenfaces projection, KNN confidence calibration
- [x] Auto-save/auto-load model persistence + REST export/import
- [x] Request-ID filter + MDC, per-stage Micrometer timers, `ModelReadyHealthIndicator`
- [x] Per-IP Bucket4j rate limiter
- [x] Experimental `OnnxDeepFeatureExtractor` scaffold (no bundled weights)
- [x] Repo hygiene: delete dead Java-3D / JAI scaffolding, `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue forms, CODEOWNERS, `.editorconfig`
- [x] CI: remove `continue-on-error`, add Checkstyle + SpotBugs + JaCoCo threshold, Dependabot, CodeQL, CycloneDX SBOM
- [x] REST integration tests via MockMvc + concurrency tests

### 2.2 — Real detection + aligned pipeline

- [ ] Bundle a permissive Haar cascade and wire a `HaarCascadeFaceDetector` that actually uses it
- [ ] Eye-landmark detector (classical) to drive `FaceAligner` without hand-picked landmarks
- [ ] Make face alignment default-on in the full pipeline
- [ ] Expand benchmark harness to report per-dataset accuracy, confusion matrices, per-stage latency percentiles

### 2.3 — Real deep-learning backend

- [ ] Finish `OnnxDeepFeatureExtractor` with a recommended free-to-use model (e.g. FaceNet-v2 or InsightFace's `buffalo_s`) under a clear licence, downloaded at first run
- [ ] Cosine-similarity scoring path optimized for 128/512-d embeddings
- [ ] GPU execution-provider selection (`facerecognition.extraction.onnx.provider: cpu|cuda|directml|coreml`)
- [ ] Reproducible LFW / CFP-FP / AgeDB-30 benchmark runs in CI (optional profile)

### 2.4 — Distributed / multi-instance

- [ ] Pluggable `ModelRepository` (S3 / GCS / PostgreSQL + pgvector)
- [ ] Stateless enrolment: embeddings in a vector store instead of in-memory identities
- [ ] Optimistic locking on enrollment / training
- [ ] Horizontal autoscaling-safe health probes

### 3.0 — SDK + demo

- [ ] Publish to Maven Central (`com.facerecognition:face-recognition`)
- [ ] Hosted demo app + public Swagger
- [ ] JavaDoc on `javadoc.io`
- [ ] Small Kotlin/Python thin-client SDK
- [ ] Performance profiling + native-image (GraalVM) build

## Non-goals

- Re-implementing OpenCV or dlib in pure Java.
- Shipping a binary blob of model weights bundled into the JAR.
- Real-time video pipelines (out of scope for a classical library — use a streaming framework).
- State-of-the-art accuracy from the classical extractors (they're in the 80–95% range on frontal aligned faces and that's by design).

## How to contribute

- **Good first issues** are tagged on GitHub. Start with docs, additional distance metrics, or extra benchmark datasets.
- **Medium tickets**: Haar cascade detector, new extractor implementations, persistence backends.
- **Hard tickets**: ONNX + deep-learning pipeline, vector store integration.

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and coding conventions.
