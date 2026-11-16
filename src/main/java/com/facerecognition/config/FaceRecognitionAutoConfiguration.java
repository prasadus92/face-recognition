package com.facerecognition.config;

import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.service.FaceClassifier;
import com.facerecognition.domain.service.FaceDetector;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.infrastructure.persistence.FileModelRepository;
import com.facerecognition.infrastructure.persistence.ModelRepository;

/**
 * Exposes the pipeline built by {@link FaceRecognitionServiceFactory} as Spring beans, configured
 * from {@code application.yml}. Each bean is {@code @ConditionalOnMissingBean}, so tests and
 * embedders can replace any piece by declaring their own bean of the same type.
 */
@Configuration(proxyBeanMethods = false)
@EnableConfigurationProperties(FaceRecognitionProperties.class)
public class FaceRecognitionAutoConfiguration {

    private static final Logger LOG = LoggerFactory.getLogger(FaceRecognitionAutoConfiguration.class);

    @Bean
    @ConditionalOnMissingBean
    public FaceDetector faceDetector(FaceRecognitionProperties props) {
        LOG.info("Initializing face detector: type={}, minFaceSize={}",
                props.getDetection().getType(), props.getDetection().getMinFaceSize());
        return FaceRecognitionServiceFactory.detector(props.getDetection());
    }

    @Bean
    @ConditionalOnMissingBean
    public FeatureExtractor featureExtractor(FaceRecognitionProperties props) {
        LOG.info("Initializing feature extractor: algorithm={}, numComponents={}",
                props.getExtraction().getAlgorithm(), props.getExtraction().getNumComponents());
        return FaceRecognitionServiceFactory.extractor(props);
    }

    @Bean
    @ConditionalOnMissingBean
    public FaceClassifier faceClassifier(FaceRecognitionProperties props) {
        return FaceRecognitionServiceFactory.classifier(props);
    }

    @Bean
    @ConditionalOnMissingBean
    public ModelRepository modelRepository(FaceRecognitionProperties props) throws IOException {
        Path storageDir = Paths.get(props.getModel().getSavePath()).toAbsolutePath().getParent();
        LOG.info("Initializing model repository at {}", storageDir);
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
        return FaceRecognitionServiceFactory.service(props, detector, extractor, classifier, modelRepository);
    }
}
