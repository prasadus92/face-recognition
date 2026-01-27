package com.facerecognition;

import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.service.FaceClassifier;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.infrastructure.classification.KNNClassifier;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Contact;
import io.swagger.v3.oas.annotations.info.Info;
import io.swagger.v3.oas.annotations.info.License;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Main Spring Boot application class for the Face Recognition REST API.
 *
 * <p>This application provides a comprehensive REST API for face recognition operations
 * including enrollment, recognition, training, and identity management.</p>
 *
 * <p>The API is documented using OpenAPI 3.0 and can be accessed via Swagger UI
 * at {@code /swagger-ui.html}.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@SpringBootApplication
@EnableConfigurationProperties
@OpenAPIDefinition(
    info = @Info(
        title = "Face Recognition API",
        version = "2.0.0",
        description = "A comprehensive REST API for face recognition operations including " +
                      "enrollment, recognition, training, and identity management.",
        contact = @Contact(
            name = "Prasad Subrahmanya",
            email = "prasadus92@gmail.com",
            url = "https://github.com/prasadus92/face-recognition"
        ),
        license = @License(
            name = "GNU General Public License v3.0",
            url = "https://www.gnu.org/licenses/gpl-3.0.html"
        )
    )
)
public class FaceRecognitionApplication {

    private static final Logger logger = LoggerFactory.getLogger(FaceRecognitionApplication.class);

    /**
     * Application entry point.
     *
     * @param args command line arguments
     */
    public static void main(String[] args) {
        logger.info("Starting Face Recognition Application...");
        SpringApplication.run(FaceRecognitionApplication.class, args);
        logger.info("Face Recognition Application started successfully");
    }

    /**
     * Creates the default feature extractor bean.
     * Uses Eigenfaces algorithm by default with 10 principal components.
     *
     * @return the FeatureExtractor instance
     */
    @Bean
    public FeatureExtractor featureExtractor() {
        logger.info("Initializing Eigenfaces feature extractor");
        return new EigenfacesExtractor(10);
    }

    /**
     * Creates the default face classifier bean.
     * Uses K-Nearest Neighbors classifier by default.
     *
     * @return the FaceClassifier instance
     */
    @Bean
    public FaceClassifier faceClassifier() {
        logger.info("Initializing KNN classifier");
        return new KNNClassifier();
    }

    /**
     * Creates the main face recognition service bean.
     *
     * @param extractor the feature extractor
     * @param classifier the face classifier
     * @return the FaceRecognitionService instance
     */
    @Bean
    public FaceRecognitionService faceRecognitionService(
            FeatureExtractor extractor,
            FaceClassifier classifier) {
        logger.info("Initializing Face Recognition Service");
        return FaceRecognitionService.builder()
                .extractor(extractor)
                .classifier(classifier)
                .config(new FaceRecognitionService.Config()
                        .setRecognitionThreshold(0.6)
                        .setDetectionConfidence(0.5)
                        .setMinQuality(0.3))
                .build();
    }
}
