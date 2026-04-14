package com.facerecognition.config;

import static org.assertj.core.api.Assertions.assertThat;

import java.io.IOException;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.config.FaceRecognitionProperties.DetectorType;
import com.facerecognition.config.FaceRecognitionProperties.ExtractorType;
import com.facerecognition.domain.service.FaceClassifier;
import com.facerecognition.domain.service.FaceClassifier.DistanceMetric;
import com.facerecognition.domain.service.FaceDetector;
import com.facerecognition.domain.service.FeatureExtractor;
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
 * Verifies that {@link FaceRecognitionAutoConfiguration} wires a fully-working
 * bean graph from {@code FaceRecognitionProperties} under every meaningful
 * combination of algorithm, classifier and detector, and that user-provided
 * beans override the defaults via {@code @ConditionalOnMissingBean}.
 */
@DisplayName("FaceRecognitionAutoConfiguration")
class FaceRecognitionAutoConfigurationTest {

    private final ApplicationContextRunner runner = new ApplicationContextRunner()
            .withConfiguration(AutoConfigurations.of(FaceRecognitionAutoConfiguration.class));

    @Test
    @DisplayName("default wiring produces Eigenfaces + KNN + composite detector + service")
    void defaultsWireUp() {
        runner.run(ctx -> {
            assertThat(ctx).hasSingleBean(FaceRecognitionProperties.class);
            assertThat(ctx).hasSingleBean(FeatureExtractor.class);
            assertThat(ctx).hasSingleBean(FaceDetector.class);
            assertThat(ctx).hasSingleBean(FaceClassifier.class);
            assertThat(ctx).hasSingleBean(FaceRecognitionService.class);
            assertThat(ctx).hasSingleBean(ModelRepository.class);

            assertThat(ctx.getBean(FeatureExtractor.class)).isInstanceOf(EigenfacesExtractor.class);
            assertThat(ctx.getBean(FaceDetector.class)).isInstanceOf(CompositeFaceDetector.class);
            assertThat(ctx.getBean(FaceClassifier.class)).isInstanceOf(KNNClassifier.class);
            assertThat(ctx.getBean(ModelRepository.class)).isInstanceOf(FileModelRepository.class);

            FaceRecognitionService service = ctx.getBean(FaceRecognitionService.class);
            assertThat(service.getDetector()).isNotNull();
            assertThat(service.getExtractor().getAlgorithmName()).isEqualTo("Eigenfaces");
            assertThat(service.getClassifier().getName()).isEqualTo("KNN");
        });
    }

    @Test
    @DisplayName("algorithm=LBPH switches the extractor bean to LBPHExtractor with configured grid")
    void lbphAlgorithm() {
        runner.withPropertyValues(
                "facerecognition.extraction.algorithm=LBPH",
                "facerecognition.extraction.lbph.grid-x=4",
                "facerecognition.extraction.lbph.grid-y=4",
                "facerecognition.extraction.lbph.radius=1",
                "facerecognition.extraction.lbph.neighbors=8"
        ).run(ctx -> {
            FeatureExtractor extractor = ctx.getBean(FeatureExtractor.class);
            assertThat(extractor).isInstanceOf(LBPHExtractor.class);
            assertThat(extractor.getAlgorithmName()).isEqualTo("LBPH");
            // 4*4 cells * 256 LBP bins == feature dimension
            assertThat(extractor.getFeatureDimension()).isEqualTo(4 * 4 * 256);
        });
    }

    @Test
    @DisplayName("algorithm=FISHERFACES switches the extractor bean to FisherfacesExtractor")
    void fisherfacesAlgorithm() {
        runner.withPropertyValues(
                "facerecognition.extraction.algorithm=FISHERFACES",
                "facerecognition.extraction.num-components=12"
        ).run(ctx -> {
            FeatureExtractor extractor = ctx.getBean(FeatureExtractor.class);
            assertThat(extractor).isInstanceOf(FisherfacesExtractor.class);
        });
    }

    @Test
    @DisplayName("algorithm=ONNX switches to the deep scaffold but stays untrained without a model")
    void onnxScaffold() {
        runner.withPropertyValues(
                "facerecognition.extraction.algorithm=ONNX",
                "facerecognition.extraction.onnx.embedding-dimension=256"
        ).run(ctx -> {
            FeatureExtractor extractor = ctx.getBean(FeatureExtractor.class);
            assertThat(extractor).isInstanceOf(OnnxDeepFeatureExtractor.class);
            assertThat(extractor.getFeatureDimension()).isEqualTo(256);
            assertThat(extractor.isTrained()).isFalse();
        });
    }

    @Test
    @DisplayName("detection.type=VIOLA_JONES swaps the detector for the standalone implementation")
    void violaJonesDetector() {
        runner.withPropertyValues("facerecognition.detection.type=VIOLA_JONES").run(ctx -> {
            assertThat(ctx.getBean(FaceDetector.class)).isInstanceOf(ViolaJonesFaceDetector.class);
        });
    }

    @Test
    @DisplayName("detection.type=SKIN_COLOR swaps the detector for the skin-colour heuristic")
    void skinColorDetector() {
        runner.withPropertyValues("facerecognition.detection.type=SKIN_COLOR").run(ctx -> {
            assertThat(ctx.getBean(FaceDetector.class)).isInstanceOf(SkinColorDetector.class);
        });
    }

    @Test
    @DisplayName("distance metric is propagated from properties to the classifier")
    void distanceMetricPropagation() {
        runner.withPropertyValues(
                "facerecognition.classification.distance-metric=COSINE",
                "facerecognition.classification.k-neighbors=5"
        ).run(ctx -> {
            FaceClassifier classifier = ctx.getBean(FaceClassifier.class);
            assertThat(classifier.getDistanceMetric()).isEqualTo(DistanceMetric.COSINE);
        });
    }

    @Test
    @DisplayName("a user-provided FaceDetector bean takes precedence over the default")
    void userBeanOverridesDefault() {
        runner.withUserConfiguration(UserDetectorConfig.class).run(ctx -> {
            FaceDetector detector = ctx.getBean(FaceDetector.class);
            assertThat(detector).isInstanceOf(ViolaJonesFaceDetector.class);
            // Only one bean of the type — our override, not the composite default.
            assertThat(ctx.getBeansOfType(FaceDetector.class)).hasSize(1);
        });
    }

    @Test
    @DisplayName("properties object is populated with the full nested schema")
    void propertiesPopulated() {
        runner.withPropertyValues(
                "facerecognition.detection.type=VIOLA_JONES",
                "facerecognition.extraction.algorithm=LBPH",
                "facerecognition.recognition.threshold=0.82",
                "facerecognition.ratelimit.requests-per-minute=7",
                "facerecognition.security.api-key=secret"
        ).run(ctx -> {
            FaceRecognitionProperties props = ctx.getBean(FaceRecognitionProperties.class);
            assertThat(props.getDetection().getType()).isEqualTo(DetectorType.VIOLA_JONES);
            assertThat(props.getExtraction().getAlgorithm()).isEqualTo(ExtractorType.LBPH);
            assertThat(props.getRecognition().getThreshold()).isEqualTo(0.82);
            assertThat(props.getRatelimit().getRequestsPerMinute()).isEqualTo(7);
            assertThat(props.getSecurity().getApiKey()).isEqualTo("secret");
        });
    }

    @Configuration
    static class UserDetectorConfig {
        @Bean
        FaceDetector userDetector() {
            return new ViolaJonesFaceDetector();
        }
    }

    /** Silences unused-warning on the IOException declared by the repo factory. */
    @SuppressWarnings("unused")
    private static void suppressChecked() throws IOException {
    }
}
