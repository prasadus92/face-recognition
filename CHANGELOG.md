# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Clean architecture restructure with domain/application/infrastructure layers
- New domain models: FaceImage, FaceRegion, FaceLandmarks, FeatureVector, Identity, RecognitionResult
- Fisherfaces (LDA) feature extractor
- LBPH (Local Binary Pattern Histogram) feature extractor
- KNN classifier with multiple distance metrics
- FaceRecognitionService for orchestrating the recognition pipeline
- Comprehensive unit tests for domain models and extractors
- Quality metrics for face images (brightness, contrast, sharpness)
- Service configuration options
- Processing metrics for recognition results
- Multiple distance metric support (Euclidean, Cosine, Manhattan, Chi-Square)

### Changed
- Restructured project to follow clean architecture principles
- Improved Eigenfaces implementation with better documentation
- Enhanced README with comprehensive documentation
- Updated project documentation structure

### Improved
- Code documentation with comprehensive Javadoc
- Error handling and validation
- Feature vector operations and comparisons

## [1.0.0] - 2024-01-15

### Added
- Initial release
- Eigenfaces (PCA) algorithm implementation
- Two-Stage Classification and Detection (TSCD)
- Swing-based GUI interface
- Face detection using skin color segmentation
- k-Nearest Neighbors classification
- MySQL user authentication
- 3D feature space visualization
- Basic face browser interface
- Image loading and preprocessing

### Features
- Face recognition under expressions, occlusions, and pose variations
- Interactive training and testing
- Batch image processing
- Progress tracking for long operations

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| 2.0.0 | TBD | Clean architecture, multiple algorithms |
| 1.0.0 | 2024-01-15 | Initial release with Eigenfaces |

---

## Migration Guide

### 1.x to 2.x

The 2.x version introduces a new clean architecture. Existing code using the legacy API will continue to work, but new development should use the new architecture.

**Legacy (1.x):**
```java
TSCD eigenFaces = new TSCD();
eigenFaces.processTrainingSet(faces, progress);
double[] features = eigenFaces.getEigenFaces(picture, numVectors);
```

**New (2.x):**
```java
FaceRecognitionService service = FaceRecognitionService.builder()
    .extractor(new EigenfacesExtractor(10))
    .classifier(new KNNClassifier())
    .build();

service.enroll(faceImage, "Identity Name");
service.train();
RecognitionResult result = service.recognize(probeImage);
```

---

## Links

- [GitHub Repository](https://github.com/prasadus92/face-recognition)
- [Issue Tracker](https://github.com/prasadus92/face-recognition/issues)
- [Documentation](https://github.com/prasadus92/face-recognition/wiki)
