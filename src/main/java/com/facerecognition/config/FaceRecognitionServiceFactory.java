package com.facerecognition.config;

import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.config.FaceRecognitionProperties.DetectorType;
import com.facerecognition.domain.service.FaceClassifier;
import com.facerecognition.domain.service.FaceClassifier.ClassifierConfig;
import com.facerecognition.domain.service.FaceDetector;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.domain.service.FeatureExtractor.ExtractorConfig;
import com.facerecognition.infrastructure.classification.KNNClassifier;
import com.facerecognition.infrastructure.detection.HaarCascadeFaceDetector;
import com.facerecognition.infrastructure.detection.SkinColorDetector;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.infrastructure.extraction.FisherfacesExtractor;
import com.facerecognition.infrastructure.extraction.LBPHExtractor;
import com.facerecognition.infrastructure.extraction.OnnxDeepFeatureExtractor;
import com.facerecognition.infrastructure.persistence.ModelRepository;

import java.nio.file.Paths;

/**
 * Builds the recognition pipeline from {@link FaceRecognitionProperties}. The Spring
 * auto-configuration and the command-line tool both use it, so they cannot drift apart.
 */
public final class FaceRecognitionServiceFactory {

    private FaceRecognitionServiceFactory() {
    }

    public static FaceDetector detector(FaceRecognitionProperties.Detection props) {
        FaceDetector detector = props.getType() == DetectorType.SKIN_COLOR
                ? new SkinColorDetector()
                : new HaarCascadeFaceDetector(
                        HaarCascadeFaceDetector.loadDefaultCascade(),
                        Math.max(props.getMinFaceSize(), HaarCascadeFaceDetector.DEFAULT_MIN_FACE_SIZE),
                        props.getScaleFactor(),
                        props.getMinNeighbours());
        detector.setMinFaceSize(props.getMinFaceSize());
        return detector;
    }

    public static FeatureExtractor extractor(FaceRecognitionProperties props) {
        FaceRecognitionProperties.Extraction extraction = props.getExtraction();
        ExtractorConfig config = new ExtractorConfig()
                .setNumComponents(extraction.getNumComponents())
                .setImageWidth(props.getImage().getTargetWidth())
                .setImageHeight(props.getImage().getTargetHeight());
        return switch (extraction.getAlgorithm()) {
            case EIGENFACES -> new EigenfacesExtractor(config);
            case FISHERFACES -> new FisherfacesExtractor(config);
            case LBPH -> {
                FaceRecognitionProperties.Extraction.Lbph lbph = extraction.getLbph();
                yield new LBPHExtractor(lbph.getGridX(), lbph.getGridY(), lbph.getRadius(), lbph.getNeighbors());
            }
            case ONNX -> {
                FaceRecognitionProperties.Extraction.Onnx onnx = extraction.getOnnx();
                yield new OnnxDeepFeatureExtractor(onnx.getModelPath(), onnx.getEmbeddingDimension(),
                        onnx.getInputSize(), onnx.getProvider());
            }
        };
    }

    public static FaceClassifier classifier(FaceRecognitionProperties props) {
        return new KNNClassifier(new ClassifierConfig()
                .setThreshold(props.getRecognition().getThreshold())
                .setK(props.getClassification().getKNeighbors())
                .setMetric(props.getClassification().getDistanceMetric())
                .setUseAverageFeatures(props.getClassification().isUseAverageFeatures()));
    }

    public static FaceRecognitionService.Config serviceConfig(FaceRecognitionProperties props) {
        return new FaceRecognitionService.Config()
                .setRecognitionThreshold(props.getRecognition().getThreshold())
                .setDetectionConfidence(props.getDetection().getMinConfidence())
                .setMinQuality(props.getQuality().getMinScore())
                .setAutoAlign(props.getImage().isFaceAlignment())
                .setTargetWidth(props.getImage().getTargetWidth())
                .setTargetHeight(props.getImage().getTargetHeight())
                .setHistogramEqualization(props.getImage().isHistogramEqualization())
                .setAutoSave(props.getModel().isAutoSave())
                .setAutoLoad(props.getModel().isAutoLoad())
                .setModelFileName(Paths.get(props.getModel().getSavePath()).getFileName().toString());
    }

    /**
     * @param props      pipeline settings
     * @param detector   face detector, or null for inputs that are already face crops
     * @param repository where models are saved, or null to manage model files yourself
     * @return a service wired from the given parts
     */
    public static FaceRecognitionService service(FaceRecognitionProperties props,
                                                 FaceDetector detector,
                                                 ModelRepository repository) {
        return service(props, detector, extractor(props), classifier(props), repository);
    }

    static FaceRecognitionService service(FaceRecognitionProperties props,
                                          FaceDetector detector,
                                          FeatureExtractor extractor,
                                          FaceClassifier classifier,
                                          ModelRepository repository) {
        return FaceRecognitionService.builder()
                .detector(detector)
                .extractor(extractor)
                .classifier(classifier)
                .modelRepository(repository)
                .config(serviceConfig(props))
                .build();
    }
}
