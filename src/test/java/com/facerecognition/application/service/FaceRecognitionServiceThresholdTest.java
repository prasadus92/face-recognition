package com.facerecognition.application.service;

import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.infrastructure.classification.KNNClassifier;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.testing.SyntheticFaces;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

@DisplayName("FaceRecognitionService recognition options")
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

    @Test
    @DisplayName("counts enrolled images per identity before training")
    void countsEnrolledImages() {
        FaceRecognitionService service = FaceRecognitionService.builder()
                .extractor(new EigenfacesExtractor(3))
                .classifier(new KNNClassifier())
                .config(new FaceRecognitionService.Config()
                        .setTargetWidth(SyntheticFaces.SIZE)
                        .setTargetHeight(SyntheticFaces.SIZE))
                .build();
        String ann = service.enroll(SyntheticFaces.face(0, 0), "Ann").getId();
        service.enroll(SyntheticFaces.face(0, 1), "Ann");
        String bo = service.enroll(SyntheticFaces.face(1, 0), "Bo").getId();

        assertThat(service.getEnrolledImageCount(ann)).isEqualTo(2);
        assertThat(service.getEnrolledImageCount(bo)).isEqualTo(1);
        assertThat(service.getEnrolledImageCount("nobody")).isZero();
    }

    @Test
    @DisplayName("training and soft deletes recalibrate the classifier")
    void recalibratesClassifier() {
        KNNClassifier classifier = spy(new KNNClassifier());
        FaceRecognitionService service = FaceRecognitionService.builder()
                .extractor(new EigenfacesExtractor(3))
                .classifier(classifier)
                .config(new FaceRecognitionService.Config()
                        .setTargetWidth(SyntheticFaces.SIZE)
                        .setTargetHeight(SyntheticFaces.SIZE))
                .build();
        String ann = service.enroll(SyntheticFaces.face(0, 0), "Ann").getId();
        service.enroll(SyntheticFaces.face(1, 0), "Bo");

        service.train();
        verify(classifier, times(1)).retrain();

        assertThat(service.setIdentityActive(ann, false)).isTrue();
        verify(classifier, times(2)).retrain();
        assertThat(service.getIdentities()).filteredOn(i -> i.getId().equals(ann))
                .singleElement().extracting(i -> i.isActive()).isEqualTo(false);
        assertThat(service.setIdentityActive("nobody", false)).isFalse();
    }
}
