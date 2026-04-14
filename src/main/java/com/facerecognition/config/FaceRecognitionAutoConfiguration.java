package com.facerecognition.config;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.config.FaceRecognitionProperties.DetectorType;
import com.facerecognition.config.FaceRecognitionProperties.ExtractorType;
import com.facerecognition.domain.service.FaceClassifier;
import com.facerecognition.domain.service.FaceClassifier.ClassifierConfig;
import com.facerecognition.domain.service.FaceDetector;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.domain.service.FeatureExtractor.ExtractorConfig;
import com.facerecognition.infrastructure.classification.KNNClassifier;
import com.facerecognition.infrastructure.detection.CompositeFaceDetector;
import com.facerecognition.infrastructure.detection.SkinColorDetector;
import com.facerecognition.infrastructure.detection.ViolaJonesFaceDetector;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.infrastructure.extraction.FisherfacesExtractor;
import com.facerecognition.infrastructure.extraction.LBPHExtractor;
import com.facerecognition.infrastructure.extraction.OnnxDeepFeatureExtractor;
import com.facerecognition.infrastructure.persistence.FileModelRepository;
import com.facerecognition.infrastructure.persistence.ModelRepository;

/**
 * Wires the face-recognition bean graph from {@link FaceRecognitionProperties}.
 *
 * <p>Every pluggable collaborator (detector, extractor, classifier, service,
 * model repository) is constructed here and only here, so operators can switch
 * algorithms and tuning by editing {@code application.yml} without recompiling.
 * Each bean is marked {@code @ConditionalOnMissingBean}, so tests and embedders
 * can still override any piece by declaring their own bean of the same type.</p>
 */
@Configuration(proxyBeanMethods = false)
@EnableConfigurationProperties(FaceRecognitionProperties.class)
public class FaceRecognitionAutoConfiguration {

    private static final Logger log = LoggerFactory.getLogger(FaceRecognitionAutoConfiguration.class);

    @Bean
    @ConditionalOnMissingBean
    public FaceDetector faceDetector(FaceRecognitionProperties props) {
        DetectorType type = props.getDetection().getType();
        int minFaceSize = props.getDetection().getMinFaceSize();
        log.info("Initializing face detector: type={}, minFaceSize={}", type, minFaceSize);
        FaceDetector detector;
        switch (type) {
            case VIOLA_JONES:
                detector = new ViolaJonesFaceDetector();
                break;
            case SKIN_COLOR:
                detector = new SkinColorDetector();
                break;
            case COMPOSITE:
            default:
                CompositeFaceDetector composite = new CompositeFaceDetector();
                composite.addDetector(new ViolaJonesFaceDetector(), 1.0);
                composite.addDetector(new SkinColorDetector(), 0.7);
                detector = composite;
                break;
        }
        detector.setMinFaceSize(minFaceSize);
        return detector;
    }

    @Bean
    @ConditionalOnMissingBean
    public FeatureExtractor featureExtractor(FaceRecognitionProperties props) {
        ExtractorType algorithm = props.getExtraction().getAlgorithm();
        int numComponents = props.getExtraction().getNumComponents();
        int width = props.getImage().getTargetWidth();
        int height = props.getImage().getTargetHeight();

        log.info("Initializing feature extractor: algorithm={}, numComponents={}, targetSize={}x{}",
                algorithm, numComponents, width, height);

        ExtractorConfig config = new ExtractorConfig()
                .setNumComponents(numComponents)
                .setImageWidth(width)
                .setImageHeight(height);

        switch (algorithm) {
            case FISHERFACES:
                return new FisherfacesExtractor(config);
            case LBPH:
                FaceRecognitionProperties.Extraction.Lbph lbph = props.getExtraction().getLbph();
                return new LBPHExtractor(lbph.getGridX(), lbph.getGridY(), lbph.getRadius(), lbph.getNeighbors());
            case ONNX:
                FaceRecognitionProperties.Extraction.Onnx onnx = props.getExtraction().getOnnx();
                return new OnnxDeepFeatureExtractor(onnx.getModelPath(), onnx.getEmbeddingDimension(),
                        onnx.getInputSize(), onnx.getProvider());
            case EIGENFACES:
            default:
                return new EigenfacesExtractor(config);
        }
    }

    @Bean
    @ConditionalOnMissingBean
    public FaceClassifier faceClassifier(FaceRecognitionProperties props) {
        ClassifierConfig config = new ClassifierConfig()
                .setThreshold(props.getRecognition().getThreshold())
                .setK(props.getClassification().getKNeighbors())
                .setMetric(props.getClassification().getDistanceMetric())
                .setUseAverageFeatures(props.getClassification().isUseAverageFeatures());

        log.info("Initializing classifier: KNN k={} metric={}",
                config.getK(), config.getMetric());
        return new KNNClassifier(config);
    }

    @Bean
    @ConditionalOnMissingBean
    public ModelRepository modelRepository(FaceRecognitionProperties props) throws IOException {
        Path savePath = Paths.get(props.getModel().getSavePath()).toAbsolutePath();
        Path storageDir = savePath.getParent();
        if (storageDir == null) {
            storageDir = Paths.get(".").toAbsolutePath();
        }
        Files.createDirectories(storageDir);
        log.info("Initializing model repository at {}", storageDir);
        return new FileModelRepository(storageDir);
    }

    @Bean
    @ConditionalOnMissingBean
    public FaceRecognitionService faceRecognitionService(
            FaceRecognitionProperties props,
            FaceDetector detector,
            FeatureExtractor extractor,
            FaceClassifier classifier,
            ModelRepository modelRepository) {

        FaceRecognitionService.Config config = new FaceRecognitionService.Config()
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

        log.info("Initializing FaceRecognitionService with extractor={} classifier={}",
                extractor.getAlgorithmName(), classifier.getName());

        return FaceRecognitionService.builder()
                .detector(detector)
                .extractor(extractor)
                .classifier(classifier)
                .modelRepository(modelRepository)
                .config(config)
                .build();
    }
}
