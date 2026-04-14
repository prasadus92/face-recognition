package com.facerecognition.api.rest.health;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.boot.actuate.health.Health;
import org.springframework.boot.actuate.health.Status;

import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.service.FaceClassifier;
import com.facerecognition.domain.service.FeatureExtractor;

@DisplayName("ModelReadyHealthIndicator")
class ModelReadyHealthIndicatorTest {

    @Test
    @DisplayName("is UP when the service reports it is trained")
    void upWhenTrained() {
        FaceRecognitionService service = mock(FaceRecognitionService.class);
        FeatureExtractor extractor = mock(FeatureExtractor.class);
        FaceClassifier classifier = mock(FaceClassifier.class);
        when(service.isTrained()).thenReturn(true);
        when(service.getIdentityCount()).thenReturn(42);
        when(service.getExtractor()).thenReturn(extractor);
        when(service.getClassifier()).thenReturn(classifier);
        when(extractor.getAlgorithmName()).thenReturn("Eigenfaces");
        when(classifier.getName()).thenReturn("KNN");

        Health health = new ModelReadyHealthIndicator(service).health();

        assertThat(health.getStatus()).isEqualTo(Status.UP);
        assertThat(health.getDetails())
                .containsEntry("enrolledIdentities", 42)
                .containsEntry("extractor", "Eigenfaces")
                .containsEntry("classifier", "KNN");
    }

    @Test
    @DisplayName("is OUT_OF_SERVICE and explains why when no identities are enrolled")
    void outOfServiceNoIdentities() {
        FaceRecognitionService service = mock(FaceRecognitionService.class);
        when(service.isTrained()).thenReturn(false);
        when(service.getIdentityCount()).thenReturn(0);

        Health health = new ModelReadyHealthIndicator(service).health();

        assertThat(health.getStatus()).isEqualTo(Status.OUT_OF_SERVICE);
        assertThat(health.getDetails())
                .containsEntry("enrolledIdentities", 0)
                .containsEntry("reason", "No identities enrolled yet");
    }

    @Test
    @DisplayName("is OUT_OF_SERVICE when identities exist but the model is not yet trained")
    void outOfServiceNeedsTraining() {
        FaceRecognitionService service = mock(FaceRecognitionService.class);
        when(service.isTrained()).thenReturn(false);
        when(service.getIdentityCount()).thenReturn(5);

        Health health = new ModelReadyHealthIndicator(service).health();

        assertThat(health.getStatus()).isEqualTo(Status.OUT_OF_SERVICE);
        assertThat(health.getDetails().get("reason")).isEqualTo("Identities enrolled but model not yet trained");
    }
}
