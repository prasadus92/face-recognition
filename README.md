<h1 align="center">Face Recognition</h1>

<p align="center">
  <strong>A classical face-recognition library for the JVM — Eigenfaces, Fisherfaces, LBPH — wrapped in a Spring Boot REST API and a Picocli CLI.</strong>
</p>

<p align="center">
  <a href="https://github.com/prasadus92/face-recognition/actions/workflows/build.yml">
    <img src="https://github.com/prasadus92/face-recognition/actions/workflows/build.yml/badge.svg?branch=master" alt="Build Status"/>
  </a>
  <a href="https://codecov.io/gh/prasadus92/face-recognition">
    <img src="https://codecov.io/gh/prasadus92/face-recognition/branch/master/graph/badge.svg" alt="Code Coverage"/>
  </a>
  <img src="https://img.shields.io/badge/java-17%2B-blue.svg" alt="Java 17+"/>
  <img src="https://img.shields.io/badge/spring--boot-3.2-brightgreen.svg" alt="Spring Boot 3.2"/>
  <a href="License.txt">
    <img src="https://img.shields.io/badge/license-GPL--3.0-blue.svg" alt="License"/>
  </a>
  <a href="https://github.com/prasadus92/face-recognition/stargazers">
    <img src="https://img.shields.io/github/stars/prasadus92/face-recognition?style=social" alt="Stars"/>
  </a>
</p>

<p align="center">
  <a href="#status">Status</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#algorithms">Algorithms</a> ·
  <a href="#rest-api">REST API</a> ·
  <a href="#cli">CLI</a> ·
  <a href="#benchmarks">Benchmarks</a> ·
  <a href="#contributing">Contributing</a>
</p>

---

## Status

> **Actively maintained.** Originally released in 2014 as a university project (`TSCD` + Swing GUI); fully rewritten for v2 as a service-shaped library with clean-architecture layering, a REST/CLI surface, and a classical-vision pipeline. v2 is **not yet** published to Maven Central — install from source (see below). See [ROADMAP.md](ROADMAP.md) for what is landed vs. planned and [CHANGELOG.md](CHANGELOG.md) for recent changes.

This repository is a learning-friendly reference implementation of the classical face-recognition pipeline — not a competitor to `dlib`, OpenCV DNN, or modern CNN-based libraries. If you need state-of-the-art accuracy, bring a real model via the [Deep-learning backend](#deep-learning-backend-experimental).

## Features

- **Three classical feature extractors** — Eigenfaces (PCA), Fisherfaces (LDA), LBPH
- **Pluggable** `FeatureExtractor` / `FaceClassifier` / `FaceDetector` interfaces
- **Clean architecture** — `domain` / `application` / `infrastructure` / `api` layers
- **REST API** with OpenAPI 3 / Swagger UI, request correlation IDs, validation, rate limiting, Prometheus metrics
- **CLI** (Picocli) for `enroll`, `train`, `recognize`, `serve`, `benchmark`
- **Model persistence** — auto-save/auto-load + REST export/import
- **Docker image** — multi-stage build, non-root user, container-aware JVM
- **Experimental ONNX backend** scaffold for FaceNet/ArcFace-style embeddings (bring your own weights)

## Status matrix

| Capability | State | Notes |
|---|---|---|
| Eigenfaces extractor | Stable | PCA via JAMA |
| Fisherfaces extractor | Stable | LDA on top of PCA |
| LBPH extractor | Stable | Uniform LBP histograms, configurable grid |
| KNN classifier | Stable | Euclidean / Cosine / Manhattan / Chi-square |
| Haar cascade face detector | Experimental | Vendored stub; real cascade data required for production |
| Face aligner (eye-centered affine) | Stable (opt-in) | Enabled via `facerecognition.image.face-alignment` |
| Model persistence | Stable | `FileModelRepository` + `TrainedModel` |
| REST + OpenAPI | Stable | `/api/v1/*` + `/swagger-ui.html` |
| Prometheus / health / metrics | Stable | Micrometer + custom `ModelReadyHealthIndicator` |
| Rate limiting | Stable | Per-IP token bucket (Bucket4j) |
| CLI (picocli) | Stable | `serve`, `recognize`, `enroll`, `train`, `benchmark` |
| ONNX deep-learning extractor | Experimental scaffold | Bring your own model weights |
| Published Maven artifact | Planned | Not yet on Central |

## Quick Start

### Prerequisites

- JDK 17+
- Maven 3.9+

### Build

```bash
git clone https://github.com/prasadus92/face-recognition.git
cd face-recognition
mvn clean package
```

The build produces two jars in `target/`:
- `face-recognition-<version>.jar` — library jar
- `face-recognition-<version>-exec.jar` — Spring Boot executable (REST + CLI)

### Run the REST API

```bash
java -jar target/face-recognition-*-exec.jar
# → http://localhost:8080/swagger-ui.html
# → http://localhost:8080/actuator/health
```

### Run the CLI

```bash
java -jar target/face-recognition-*-exec.jar --help
java -jar target/face-recognition-*-exec.jar enroll --image john.jpg --name "John Doe"
java -jar target/face-recognition-*-exec.jar train
java -jar target/face-recognition-*-exec.jar recognize --image unknown.jpg
```

### Run with Docker

```bash
docker build -t face-recognition:latest .
docker run --rm -p 8080:8080 -v "$PWD/data:/app/data" face-recognition:latest
```

### Use as a library

```java
import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.infrastructure.classification.KNNClassifier;
import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.RecognitionResult;

import java.io.File;

FaceRecognitionService service = FaceRecognitionService.builder()
    .extractor(new EigenfacesExtractor(10))
    .classifier(new KNNClassifier())
    .build();

service.enrollFromFile(new File("john.jpg"), "John Doe");
service.enrollFromFile(new File("jane.jpg"), "Jane Smith");
service.train();

RecognitionResult result = service.recognizeFromFile(new File("unknown.jpg"));
result.getBestMatch().ifPresent(match ->
    System.out.printf("%s (conf=%.2f)%n",
        match.getIdentity().getName(), match.getConfidence()));
```

## Algorithms

| Algorithm | Idea | Best for | Tradeoffs |
|---|---|---|---|
| **Eigenfaces (PCA)** | Project faces onto principal components of training set. | Fast, controlled environments. | Sensitive to lighting and pose. |
| **Fisherfaces (LDA)** | Maximize between-class / within-class scatter. | Varied lighting with ≥2 samples/identity. | Requires multiple labelled samples per identity. |
| **LBPH** | Concatenate histograms of local binary patterns over a grid. | Texture-based matching under lighting changes. | Pose-sensitive; higher memory. |

The classical extractors live under `com.facerecognition.infrastructure.extraction` and are selected declaratively via `application.yml` (`facerecognition.extraction.algorithm = eigenfaces | fisherfaces | lbph | onnx`) or programmatically via the service builder.

### Deep-learning backend (experimental)

An `OnnxDeepFeatureExtractor` scaffold is included for running modern embedding networks (FaceNet, ArcFace, etc.) via ONNX Runtime. **It ships without model weights** — you must supply your own (`.onnx`) and configure the model path in `application.yml`. See `docs/onnx.md` (forthcoming) for the contract.

## REST API

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/v1/enroll` | POST multipart | Register a face sample for a name |
| `/api/v1/recognize` | POST multipart | Identify a face in an uploaded image |
| `/api/v1/train` | POST | Train / retrain on all enrolled samples |
| `/api/v1/identities` | GET | List enrolled identities (paginated) |
| `/api/v1/identities/{id}` | GET / PATCH / DELETE | Read, update, soft-delete an identity |
| `/api/v1/model/status` | GET | Model state, algorithm, identity counts |
| `/api/v1/model/export` | POST | Download a serialized `TrainedModel` |
| `/api/v1/model/import` | POST multipart | Load a previously exported model |
| `/actuator/health` | GET | Liveness + custom `model-ready` indicator |
| `/actuator/prometheus` | GET | Micrometer metrics in Prometheus format |
| `/swagger-ui.html` | GET | Interactive OpenAPI 3 docs |

Every response that represents an error uses the [`ErrorResponse`](src/main/java/com/facerecognition/api/rest/dto/ErrorResponse.java) shape and carries a `traceId` (echoed from the `X-Request-ID` request header or generated if absent).

### Security and limits

- **Rate limit**: per-IP token bucket (default 60 req/min, configurable via `facerecognition.ratelimit.*`).
- **Upload limit**: `spring.servlet.multipart.max-file-size` (10 MB default).
- **CORS**: disabled by default; enable via `facerecognition.cors.*`.
- **Authentication**: no auth by default. Deploy behind an API gateway / reverse proxy, or enable the optional API-key filter (`facerecognition.security.api-key`).

## CLI

```text
Usage: face-recognition [COMMAND]
Commands:
  enroll      Enroll a face image under a name.
  train       Train or retrain the model.
  recognize   Recognize a face from an image or directory.
  benchmark   Run accuracy/performance benchmarks on a dataset.
  serve       Start the HTTP API server.
```

All commands honour `--model <path>` and `--config <application.yml>` for reproducible runs.

## Configuration

All runtime settings live in `src/main/resources/application.yml` and can be overridden via environment variables (`FACERECOGNITION_EXTRACTION_ALGORITHM=lbph`) or a custom `application.yml` passed to `--spring.config.location`. Core keys:

```yaml
facerecognition:
  detection:
    min-face-size: 30
    min-confidence: 0.5
  extraction:
    algorithm: eigenfaces      # eigenfaces | fisherfaces | lbph | onnx
    num-components: 10
    onnx:
      model-path: ""           # path to a .onnx model for the deep backend
  classification:
    algorithm: knn
    k-neighbors: 3
    distance-metric: euclidean # euclidean | cosine | manhattan | chi_square
  recognition:
    threshold: 0.6
  quality:
    min-score: 0.3
  image:
    target-width: 100
    target-height: 100
    face-alignment: true
  model:
    auto-save: true
    auto-load: true
    save-path: data/models/default.frm
  ratelimit:
    enabled: true
    requests-per-minute: 60
```

## Benchmarks

> **Honest disclaimer.** The numbers currently checked into [docs/benchmarks/](docs/benchmarks/) come from the bundled micro-dataset (`src/test/resources/datasets/mini/`) and are not directly comparable to LFW or Yale B. Classical algorithms on aligned frontal faces typically land in the 80–95% range depending on dataset; don't expect deep-learning-level accuracy.

Run the benchmark suite locally:

```bash
mvn -P benchmarks exec:java
# or via the CLI
java -jar target/face-recognition-*-exec.jar benchmark \
    --dataset src/test/resources/datasets/mini \
    --algorithm all \
    --report docs/benchmarks/mini.json
```

The benchmark harness lives under `com.facerecognition.benchmark` and reports top-1 accuracy, per-stage latency, and confusion matrices as JSON / Markdown.

## Observability

- **Logs** — structured via Logback, per-request `traceId` via MDC; file appender rolls daily with 1 GB cap.
- **Metrics** — custom Micrometer timers `facerecognition.detect`, `facerecognition.extract`, `facerecognition.match`, `facerecognition.recognize.total`; counters for recognitions / enrollments / errors.
- **Health** — `/actuator/health` exposes a custom `model-ready` component reflecting `FaceRecognitionService#isTrained()`.
- **Tracing** — propagate `X-Request-ID` from your upstream and it will appear in every log line and response body.

## Project layout

```
face-recognition/
├── src/
│   ├── main/
│   │   ├── java/com/facerecognition/
│   │   │   ├── FaceRecognitionApplication.java     # Spring Boot entry point
│   │   │   ├── config/                             # @ConfigurationProperties + bean factories
│   │   │   ├── domain/                             # Pure domain model + service interfaces
│   │   │   │   ├── model/
│   │   │   │   └── service/
│   │   │   ├── application/service/                # Orchestrator (FaceRecognitionService)
│   │   │   ├── infrastructure/
│   │   │   │   ├── classification/
│   │   │   │   ├── detection/
│   │   │   │   ├── extraction/
│   │   │   │   ├── persistence/
│   │   │   │   └── preprocessing/
│   │   │   ├── api/
│   │   │   │   ├── rest/                           # Controllers, DTOs, advice, filters
│   │   │   │   └── cli/                            # Picocli commands
│   │   │   └── benchmark/                          # Accuracy + performance harness
│   │   └── resources/
│   │       ├── application.yml
│   │       └── logback-spring.xml
│   └── test/java/com/facerecognition/              # Unit + integration tests
├── docs/
│   ├── architecture.md
│   ├── benchmarks/
│   └── onnx.md
├── .github/
│   ├── workflows/                                  # CI, CodeQL, Dependabot, release
│   ├── ISSUE_TEMPLATE/
│   └── PULL_REQUEST_TEMPLATE.md
├── CHANGELOG.md
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── ROADMAP.md
├── Dockerfile
└── pom.xml
```

## Contributing

Contributions are very welcome — bug fixes, extractors, distance metrics, deep-learning adapters, documentation. Start with [CONTRIBUTING.md](CONTRIBUTING.md), which covers the development loop, coding standards, and how to run the full test + quality-gate suite locally. Please also read the [Code of Conduct](CODE_OF_CONDUCT.md).

## Security

Report vulnerabilities privately — see [SECURITY.md](SECURITY.md). Do **not** file public issues for security bugs.

## References

This implementation builds on foundational papers:

1. Turk, M. & Pentland, A. (1991). *Eigenfaces for Recognition.*
2. Belhumeur, P. N., Hespanha, J. P. & Kriegman, D. J. (1997). *Eigenfaces vs. Fisherfaces: Recognition Using Class Specific Linear Projection.*
3. Ahonen, T., Hadid, A. & Pietikäinen, M. (2006). *Face Description with Local Binary Patterns.*
4. Schroff, F., Kalenichenko, D. & Philbin, J. (2015). *FaceNet: A Unified Embedding for Face Recognition and Clustering.*
5. Deng, J., Guo, J., Xue, N. & Zafeiriou, S. (2019). *ArcFace: Additive Angular Margin Loss for Deep Face Recognition.*

## License

Licensed under the **GNU General Public License v3.0** — see [License.txt](License.txt).

## Acknowledgements

- [JAMA](https://math.nist.gov/javanumerics/jama/) for numerical linear algebra.
- [Spring Boot](https://spring.io/projects/spring-boot), [picocli](https://picocli.info/), [Micrometer](https://micrometer.io/), [Bucket4j](https://bucket4j.com/), [springdoc-openapi](https://springdoc.org/).
- Maintainer: **[Prasad Subrahmanya](https://github.com/prasadus92)** · prasadus92@gmail.com
