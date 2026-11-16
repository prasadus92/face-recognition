package com.facerecognition.application.service;

import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.infrastructure.classification.KNNClassifier;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.testing.SyntheticFaces;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("FaceRecognitionService thresholds")
class FaceRecognitionServiceThresholdTest {

    @Test
    @DisplayName("a per-call threshold overrides the configured one")
    void perCallThreshold() {
        FaceRecognitionService service = FaceRecognitionService.builder()
                .extractor(new EigenfacesExtractor(3))
                .classifier(new KNNClassifier())
                .config(new FaceRecognitionService.Config()
                        .setRecognitionThreshold(0.0)
                        .setTargetWidth(SyntheticFaces.SIZE)
                        .setTargetHeight(SyntheticFaces.SIZE))
                .build();
        for (int person = 0; person < 2; person++) {
            for (int sample = 0; sample < 3; sample++) {
                service.enroll(SyntheticFaces.face(person, sample), "person-" + person);
            }
        }
        service.train();

        assertThat(service.recognize(SyntheticFaces.face(1, 9)).getStatus())
                .isEqualTo(RecognitionResult.Status.RECOGNIZED);
        assertThat(service.recognize(SyntheticFaces.face(1, 9), 1.0).getStatus())
                .isEqualTo(RecognitionResult.Status.UNKNOWN);
    }
}
