package com.facerecognition.application.service;

import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.infrastructure.classification.KNNClassifier;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.infrastructure.extraction.FisherfacesExtractor;
import com.facerecognition.infrastructure.extraction.LBPHExtractor;
import com.facerecognition.infrastructure.persistence.FileModelRepository;
import com.facerecognition.infrastructure.persistence.ModelRepository;
import com.facerecognition.testing.SyntheticFaces;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

import java.nio.file.Path;
import java.util.function.Supplier;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("FaceRecognitionService persistence")
class FaceRecognitionServicePersistenceTest {

    private static final int IDENTITIES = 3;
    private static final int SAMPLES = 4;

    @TempDir
    Path dir;

    private ModelRepository repository;

    @BeforeEach
    void setUp() throws Exception {
        repository = new FileModelRepository(dir);
    }

    static Stream<Supplier<FeatureExtractor>> extractors() {
        return Stream.of(
                () -> new EigenfacesExtractor(5),
                () -> new FisherfacesExtractor(2),
                LBPHExtractor::new);
    }

    @ParameterizedTest
    @MethodSource("extractors")
    @DisplayName("a saved model recognizes faces after a restart without retraining")
    void recognizesAfterRestart(Supplier<FeatureExtractor> extractor) throws Exception {
        FaceRecognitionService original = newService(extractor.get());
        enroll(original, 0, IDENTITIES);
        original.train();
        original.saveModel();

        FaceRecognitionService restarted = newService(extractor.get());

        assertThat(restarted.tryLoadSavedModel()).isTrue();
        assertThat(restarted.isTrained()).isTrue();
        for (int id = 0; id < IDENTITIES; id++) {
            RecognitionResult before = original.recognize(SyntheticFaces.face(id, 99));
            RecognitionResult after = restarted.recognize(SyntheticFaces.face(id, 99));
            assertThat(bestName(after)).isEqualTo("person-" + id).isEqualTo(bestName(before));
            assertThat(after.getConfidence()).isEqualTo(before.getConfidence());
        }
    }

    @Test
    @DisplayName("a restored model can be retrained with a new identity and keeps the old ones")
    void retrainsAfterRestore() throws Exception {
        FaceRecognitionService original = newService(new EigenfacesExtractor(5));
        enroll(original, 0, IDENTITIES);
        original.train();
        original.saveModel();

        FaceRecognitionService restarted = newService(new EigenfacesExtractor(5));
        restarted.tryLoadSavedModel();
        enroll(restarted, IDENTITIES, IDENTITIES + 1);
        restarted.train();

        for (int id = 0; id <= IDENTITIES; id++) {
            assertThat(bestName(restarted.recognize(SyntheticFaces.face(id, 99)))).isEqualTo("person-" + id);
        }
    }

    @Test
    @DisplayName("export then import keeps identity ids, external ids, status and metadata")
    void importKeepsIdentityRecords() throws Exception {
        FaceRecognitionService original = newService(new EigenfacesExtractor(5));
        enroll(original, 0, IDENTITIES);
        Identity first = original.getIdentities().stream()
                .filter(i -> i.getName().equals("person-0")).findFirst().orElseThrow();
        first.setExternalId("crm-42");
        first.setMetadata("team", "red");
        first.setActive(false);
        original.train();

        FaceRecognitionService copy = newService(new EigenfacesExtractor(5));
        copy.importModel(original.exportModel());

        Identity restored = copy.getIdentities().stream()
                .filter(i -> i.getId().equals(first.getId())).findFirst().orElseThrow();
        assertThat(restored.getName()).isEqualTo("person-0");
        assertThat(restored.getExternalId()).isEqualTo("crm-42");
        assertThat(restored.getMetadata("team")).isEqualTo("red");
        assertThat(restored.isActive()).isFalse();
        assertThat(copy.getIdentityCount()).isEqualTo(IDENTITIES);
    }

    @Test
    @DisplayName("a model saved before training restores as untrained with its samples")
    void untrainedModelRestoresUntrained() throws Exception {
        FaceRecognitionService original = newService(new EigenfacesExtractor(5));
        enroll(original, 0, IDENTITIES);
        original.saveModel();

        FaceRecognitionService restarted = newService(new EigenfacesExtractor(5));
        restarted.tryLoadSavedModel();

        assertThat(restarted.isTrained()).isFalse();
        restarted.train();
        assertThat(bestName(restarted.recognize(SyntheticFaces.face(1, 99)))).isEqualTo("person-1");
    }

    private FaceRecognitionService newService(FeatureExtractor extractor) {
        return FaceRecognitionService.builder()
                .extractor(extractor)
                .classifier(new KNNClassifier())
                .modelRepository(repository)
                .config(new FaceRecognitionService.Config()
                        .setRecognitionThreshold(0.0)
                        .setTargetWidth(SyntheticFaces.SIZE)
                        .setTargetHeight(SyntheticFaces.SIZE))
                .build();
    }

    private static void enroll(FaceRecognitionService service, int fromIdentity, int toIdentity) {
        for (int id = fromIdentity; id < toIdentity; id++) {
            for (int sample = 0; sample < SAMPLES; sample++) {
                service.enroll(SyntheticFaces.face(id, sample), "person-" + id);
            }
        }
    }

    private static String bestName(RecognitionResult result) {
        return result.getBestMatch().map(m -> m.getIdentity().getName()).orElse("unknown");
    }
}
