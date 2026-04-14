package com.facerecognition.api.rest.health;

import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.HealthIndicator;
import org.springframework.stereotype.Component;

import com.facerecognition.application.service.FaceRecognitionService;

/**
 * Spring Boot Actuator {@link HealthIndicator} that reports the face-recognition
 * pipeline's readiness. The service is {@code UP} iff a trained model is present,
 * otherwise {@code OUT_OF_SERVICE} with the number of currently enrolled
 * identities for diagnostics.
 *
 * <p>Exposed under {@code /actuator/health/modelReady} when the actuator is
 * configured with {@code management.endpoint.health.show-components=always}.</p>
 */
@Component("modelReady")
public class ModelReadyHealthIndicator implements HealthIndicator {

    private final FaceRecognitionService service;

    public ModelReadyHealthIndicator(FaceRecognitionService service) {
        this.service = service;
    }

    @Override
    public Health health() {
        int enrolled = service.getIdentityCount();
        if (service.isTrained()) {
            return Health.up()
                    .withDetail("enrolledIdentities", enrolled)
                    .withDetail("extractor", service.getExtractor().getAlgorithmName())
                    .withDetail("classifier", service.getClassifier().getName())
                    .build();
        }
        return Health.outOfService()
                .withDetail("enrolledIdentities", enrolled)
                .withDetail("reason", enrolled == 0
                        ? "No identities enrolled yet"
                        : "Identities enrolled but model not yet trained")
                .build();
    }
}
