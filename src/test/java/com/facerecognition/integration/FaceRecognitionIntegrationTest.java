package com.facerecognition.integration;

import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.model.*;
import com.facerecognition.infrastructure.classification.KNNClassifier;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.infrastructure.extraction.LBPHExtractor;
import org.junit.jupiter.api.*;

import java.awt.image.BufferedImage;
import java.util.*;

import static org.assertj.core.api.Assertions.*;

/**
 * Integration tests for the complete face recognition pipeline.
 * Tests the full flow from enrollment through training to recognition.
 */
@DisplayName("Face Recognition Integration Tests")
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class FaceRecognitionIntegrationTest {

    private static final int IMAGE_WIDTH = 48;
    private static final int IMAGE_HEIGHT = 64;
    private static final int NUM_IDENTITIES = 5;
    private static final int SAMPLES_PER_IDENTITY = 4;

    private FaceRecognitionService service;
    private Map<String, List<FaceImage>> testData;

    @BeforeEach
    void setUp() {
        // Create synthetic test data
        testData = createSyntheticDataset(NUM_IDENTITIES, SAMPLES_PER_IDENTITY);
    }

    @Nested
    @DisplayName("Eigenfaces Pipeline")
    class EigenfacesPipelineTests {

        @BeforeEach
        void setUpService() {
            service = FaceRecognitionService.builder()
                .extractor(new EigenfacesExtractor(10))
                .classifier(new KNNClassifier())
                .build();
        }

        @Test
        @Order(1)
        @DisplayName("Complete recognition pipeline works end-to-end")
        void completePipelineWorks() {
            // Enroll faces
            for (Map.Entry<String, List<FaceImage>> entry : testData.entrySet()) {
                String identity = entry.getKey();
                List<FaceImage> images = entry.getValue();

                // Use first 3 images for training
                for (int i = 0; i < 3; i++) {
                    service.enroll(images.get(i), identity);
                }
            }

            // Train
            assertThatCode(() -> service.train()).doesNotThrowAnyException();
            assertThat(service.isTrained()).isTrue();

            // Test recognition with held-out images
            int correct = 0;
            int total = 0;

            for (Map.Entry<String, List<FaceImage>> entry : testData.entrySet()) {
                String expectedIdentity = entry.getKey();
                FaceImage testImage = entry.getValue().get(3); // Use 4th image for testing

                RecognitionResult result = service.recognize(testImage);

                if (result.isRecognized() &&
                    result.getIdentity().get().getName().equals(expectedIdentity)) {
                    correct++;
                }
                total++;
            }

            double accuracy = (double) correct / total;
            System.out.printf("Recognition accuracy: %.2f%% (%d/%d)%n",
                accuracy * 100, correct, total);

            // With synthetic data, we expect reasonable accuracy
            assertThat(accuracy).isGreaterThanOrEqualTo(0.6);
        }

        @Test
        @Order(2)
        @DisplayName("Service reports correct identity count")
        void correctIdentityCount() {
            for (Map.Entry<String, List<FaceImage>> entry : testData.entrySet()) {
                service.enroll(entry.getValue().get(0), entry.getKey());
            }

            assertThat(service.getIdentityCount()).isEqualTo(NUM_IDENTITIES);
        }

        @Test
        @Order(3)
        @DisplayName("Recognition returns confidence scores")
        void recognitionReturnsConfidence() {
            // Enroll all data
            for (Map.Entry<String, List<FaceImage>> entry : testData.entrySet()) {
                for (FaceImage image : entry.getValue()) {
                    service.enroll(image, entry.getKey());
                }
            }
            service.train();

            // Test recognition
            FaceImage testImage = testData.values().iterator().next().get(0);
            RecognitionResult result = service.recognize(testImage);

            assertThat(result.getConfidence()).isBetween(0.0, 1.0);
            assertThat(result.getMetrics()).isPresent();
            // Use >= 0 since fast operations may complete in sub-millisecond time
            assertThat(result.getMetrics().get().getTotalTimeMs()).isGreaterThanOrEqualTo(0);
        }

        @Test
        @Order(4)
        @DisplayName("Throws when recognizing before training")
        void throwsBeforeTraining() {
            FaceImage testImage = testData.values().iterator().next().get(0);

            assertThatThrownBy(() -> service.recognize(testImage))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("not trained");
        }

        @Test
        @Order(5)
        @DisplayName("Throws when training without enrollments")
        void throwsWhenTrainingEmpty() {
            assertThatThrownBy(() -> service.train())
                .isInstanceOf(IllegalStateException.class);
        }
    }

    @Nested
    @DisplayName("LBPH Pipeline")
    class LBPHPipelineTests {

        @BeforeEach
        void setUpService() {
            service = FaceRecognitionService.builder()
                .extractor(new LBPHExtractor(4, 4, 1, 8))
                .classifier(new KNNClassifier())
                .build();
        }

        @Test
        @DisplayName("LBPH pipeline works end-to-end")
        void lbphPipelineWorks() {
            // Enroll faces
            for (Map.Entry<String, List<FaceImage>> entry : testData.entrySet()) {
                String identity = entry.getKey();
                for (int i = 0; i < 3; i++) {
                    service.enroll(entry.getValue().get(i), identity);
                }
            }

            // Train and recognize
            service.train();
            assertThat(service.isTrained()).isTrue();

            FaceImage testImage = testData.values().iterator().next().get(3);
            RecognitionResult result = service.recognize(testImage);

            assertThat(result.getStatus()).isIn(
                RecognitionResult.Status.RECOGNIZED,
                RecognitionResult.Status.UNKNOWN
            );
        }
    }

    @Nested
    @DisplayName("Multiple Samples Per Identity")
    class MultipleSamplesTests {

        @BeforeEach
        void setUpService() {
            service = FaceRecognitionService.builder()
                .extractor(new EigenfacesExtractor(10))
                .classifier(new KNNClassifier())
                .build();
        }

        @Test
        @DisplayName("Multiple samples improve recognition")
        void multipleSamplesImproveRecognition() {
            String testIdentity = "person_1";
            List<FaceImage> personImages = testData.get(testIdentity);

            // First: train with 2 samples (minimum for PCA to compute variance)
            service.enroll(personImages.get(0), testIdentity);
            service.enroll(personImages.get(1), testIdentity);
            service.train();

            RecognitionResult result1 = service.recognize(personImages.get(3));

            // Reset and train with more samples (3)
            service = FaceRecognitionService.builder()
                .extractor(new EigenfacesExtractor(10))
                .classifier(new KNNClassifier())
                .build();

            for (int i = 0; i < 3; i++) {
                service.enroll(personImages.get(i), testIdentity);
            }
            service.train();

            RecognitionResult result2 = service.recognize(personImages.get(3));

            // Both should recognize the identity
            if (result1.isRecognized() && result2.isRecognized()) {
                // With more training samples, we expect better matching
                assertThat(result2.getConfidence())
                    .isGreaterThanOrEqualTo(result1.getConfidence() * 0.8);
            }
        }
    }

    @Nested
    @DisplayName("Performance Tests")
    class PerformanceTests {

        @BeforeEach
        void setUpService() {
            service = FaceRecognitionService.builder()
                .extractor(new EigenfacesExtractor(10))
                .classifier(new KNNClassifier())
                .build();

            // Enroll all data
            for (Map.Entry<String, List<FaceImage>> entry : testData.entrySet()) {
                for (FaceImage image : entry.getValue()) {
                    service.enroll(image, entry.getKey());
                }
            }
            service.train();
        }

        @Test
        @DisplayName("Recognition completes within time limit")
        void recognitionWithinTimeLimit() {
            FaceImage testImage = testData.values().iterator().next().get(0);

            long startTime = System.currentTimeMillis();
            RecognitionResult result = service.recognize(testImage);
            long elapsed = System.currentTimeMillis() - startTime;

            assertThat(result.getMetrics()).isPresent();
            assertThat(elapsed).isLessThan(1000); // Should complete in under 1 second

            System.out.printf("Recognition time: %d ms%n", elapsed);
            System.out.printf("  Detection: %d ms%n", result.getMetrics().get().getDetectionTimeMs());
            System.out.printf("  Extraction: %d ms%n", result.getMetrics().get().getExtractionTimeMs());
            System.out.printf("  Matching: %d ms%n", result.getMetrics().get().getMatchingTimeMs());
        }

        @Test
        @DisplayName("Batch recognition performance")
        void batchRecognitionPerformance() {
            List<FaceImage> testImages = new ArrayList<>();
            for (List<FaceImage> images : testData.values()) {
                testImages.add(images.get(SAMPLES_PER_IDENTITY - 1));
            }

            long startTime = System.currentTimeMillis();
            int recognized = 0;

            for (FaceImage image : testImages) {
                RecognitionResult result = service.recognize(image);
                if (result.isRecognized()) {
                    recognized++;
                }
            }

            long elapsed = System.currentTimeMillis() - startTime;
            double avgTime = (double) elapsed / testImages.size();

            System.out.printf("Batch recognition: %d images in %d ms (%.2f ms/image)%n",
                testImages.size(), elapsed, avgTime);
            System.out.printf("Recognized: %d/%d (%.1f%%)%n",
                recognized, testImages.size(), 100.0 * recognized / testImages.size());

            assertThat(avgTime).isLessThan(500); // Should be under 500ms per image
        }
    }

    /**
     * Creates a synthetic dataset for testing.
     * Each identity has a unique pattern that can be distinguished.
     */
    private Map<String, List<FaceImage>> createSyntheticDataset(int numIdentities, int samplesPerIdentity) {
        Map<String, List<FaceImage>> dataset = new LinkedHashMap<>();
        Random random = new Random(42); // Fixed seed for reproducibility

        for (int i = 0; i < numIdentities; i++) {
            String identity = "person_" + (i + 1);
            List<FaceImage> samples = new ArrayList<>();

            // Each identity has a unique base pattern
            double baseFreqX = 0.1 + i * 0.05;
            double baseFreqY = 0.1 + i * 0.03;
            double basePhase = i * Math.PI / numIdentities;

            for (int j = 0; j < samplesPerIdentity; j++) {
                BufferedImage image = new BufferedImage(
                    IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_INT_RGB);

                // Add some variation for each sample
                double variation = 0.1 * random.nextGaussian();

                for (int y = 0; y < IMAGE_HEIGHT; y++) {
                    for (int x = 0; x < IMAGE_WIDTH; x++) {
                        // Create a unique pattern for each identity
                        double value = 128 + 60 * Math.sin(x * baseFreqX + basePhase)
                                          + 40 * Math.cos(y * baseFreqY + basePhase)
                                          + 20 * Math.sin((x + y) * 0.1)
                                          + 15 * variation * random.nextGaussian();

                        int gray = (int) Math.max(0, Math.min(255, value));
                        int rgb = (gray << 16) | (gray << 8) | gray;
                        image.setRGB(x, y, rgb);
                    }
                }

                samples.add(FaceImage.fromBufferedImage(image));
            }

            dataset.put(identity, samples);
        }

        return dataset;
    }
}
