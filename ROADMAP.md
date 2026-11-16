# Roadmap

What works today, and what I would build next. Issues and pull requests on any item are welcome.

## Works today

- Eigenfaces, Fisherfaces and LBPH behind one `FeatureExtractor` interface, measured on AT&T/ORL by a script anyone can rerun ([results](docs/benchmarks/orl.md)).
- A pure-Java Haar face detector that matches OpenCV 4.x on the bundled frontal-face cascade.
- Nearest-neighbor matching with confidence calibrated on the gallery, so the threshold rejects strangers.
- One executable jar: a Spring Boot REST API (OpenAPI, Actuator, Prometheus, rate limiting, optional API key) and a CLI (`train`, `enroll`, `recognize`, `detect`, `benchmark`).
- Model files that carry the fitted extractor and the training crops, read back behind a deserialization allowlist.
- A Docker image built and smoke-tested in CI.

## Next

- **A face-embedding baseline.** An ONNX Runtime extractor for a modern embedding network, behind `FeatureExtractor`, benchmarked next to the classical methods. The open question is a model whose license allows redistribution; most public ArcFace weights are for non-commercial research only.
- **Store the pipeline settings in the model file.** The distance metric and whether detection ran at enrollment should travel with the model, so `recognize` cannot silently use a different setup from `train`.
- **Landmark-based alignment.** Locate the eyes inside the detected face and level them before cropping; today the crop is the padded detector box.
- **More datasets.** Extended Yale B for lighting and LFW for photos taken in the wild, both through `benchmark`.
- **Faster detection.** `int` integral images, buffer reuse across pyramid levels and OpenCV's skip after an early rejection.
- **Publish to Maven Central** so the library can be used without the server.
