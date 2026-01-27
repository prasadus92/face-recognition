package com.facerecognition.unit.infrastructure;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import org.junit.jupiter.api.*;

import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for EigenfacesExtractor.
 */
@DisplayName("EigenfacesExtractor Tests")
class EigenfacesExtractorTest {

    private static final int IMAGE_WIDTH = 48;
    private static final int IMAGE_HEIGHT = 64;
    private static final int NUM_COMPONENTS = 5;

    private EigenfacesExtractor extractor;
    private List<FaceImage> trainingImages;

    @BeforeEach
    void setUp() {
        FeatureExtractor.ExtractorConfig config = new FeatureExtractor.ExtractorConfig()
            .setNumComponents(NUM_COMPONENTS)
            .setImageWidth(IMAGE_WIDTH)
            .setImageHeight(IMAGE_HEIGHT);

        extractor = new EigenfacesExtractor(config);
        trainingImages = createTestImages(10);
    }

    private List<FaceImage> createTestImages(int count) {
        List<FaceImage> images = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            BufferedImage img = new BufferedImage(IMAGE_WIDTH, IMAGE_HEIGHT, BufferedImage.TYPE_INT_RGB);
            // Create synthetic face-like patterns
            for (int y = 0; y < IMAGE_HEIGHT; y++) {
                for (int x = 0; x < IMAGE_WIDTH; x++) {
                    int gray = (int) (128 + 50 * Math.sin(x * 0.2 + i) * Math.cos(y * 0.2 + i));
                    gray = Math.max(0, Math.min(255, gray));
                    int rgb = (gray << 16) | (gray << 8) | gray;
                    img.setRGB(x, y, rgb);
                }
            }
            images.add(FaceImage.fromBufferedImage(img));
        }
        return images;
    }

    @Nested
    @DisplayName("Training")
    class TrainingTests {

        @Test
        @DisplayName("Trains successfully with valid data")
        void trainsSuccessfully() {
            assertDoesNotThrow(() -> extractor.train(trainingImages, null));
            assertTrue(extractor.isTrained());
        }

        @Test
        @DisplayName("Not trained before training")
        void notTrainedInitially() {
            assertFalse(extractor.isTrained());
        }

        @Test
        @DisplayName("Throws on empty training set")
        void throwsOnEmptyTrainingSet() {
            assertThrows(IllegalArgumentException.class,
                () -> extractor.train(new ArrayList<>(), null));
        }

        @Test
        @DisplayName("Throws on null training set")
        void throwsOnNullTrainingSet() {
            assertThrows(IllegalArgumentException.class,
                () -> extractor.train(null, null));
        }

        @Test
        @DisplayName("Throws when training twice without reset")
        void throwsOnDoubleTrain() {
            extractor.train(trainingImages, null);
            assertThrows(IllegalStateException.class,
                () -> extractor.train(trainingImages, null));
        }

        @Test
        @DisplayName("Can train after reset")
        void canTrainAfterReset() {
            extractor.train(trainingImages, null);
            extractor.reset();
            assertFalse(extractor.isTrained());
            assertDoesNotThrow(() -> extractor.train(trainingImages, null));
        }
    }

    @Nested
    @DisplayName("Feature Extraction")
    class ExtractionTests {

        @BeforeEach
        void trainExtractor() {
            extractor.train(trainingImages, null);
        }

        @Test
        @DisplayName("Extracts features with correct dimension")
        void extractsCorrectDimension() {
            FaceImage probe = trainingImages.get(0);
            FeatureVector features = extractor.extract(probe);

            assertNotNull(features);
            assertTrue(features.getDimension() <= NUM_COMPONENTS);
        }

        @Test
        @DisplayName("Extracted features have algorithm name")
        void hasAlgorithmName() {
            FeatureVector features = extractor.extract(trainingImages.get(0));
            assertEquals("Eigenfaces", features.getAlgorithmName());
        }

        @Test
        @DisplayName("Similar images have similar features")
        void similarImagesSimilarFeatures() {
            // Extract features from the same image twice
            FaceImage img = trainingImages.get(0);
            FeatureVector f1 = extractor.extract(img);
            FeatureVector f2 = extractor.extract(img);

            double distance = f1.euclideanDistance(f2);
            assertEquals(0.0, distance, 1e-6);
        }

        @Test
        @DisplayName("Different images have different features")
        void differentImagesDifferentFeatures() {
            FeatureVector f1 = extractor.extract(trainingImages.get(0));
            FeatureVector f2 = extractor.extract(trainingImages.get(5));

            double distance = f1.euclideanDistance(f2);
            assertTrue(distance > 0.01);
        }

        @Test
        @DisplayName("Throws when extracting before training")
        void throwsBeforeTraining() {
            EigenfacesExtractor untrained = new EigenfacesExtractor();
            assertThrows(IllegalStateException.class,
                () -> untrained.extract(trainingImages.get(0)));
        }
    }

    @Nested
    @DisplayName("Model Properties")
    class ModelPropertiesTests {

        @BeforeEach
        void trainExtractor() {
            extractor.train(trainingImages, null);
        }

        @Test
        @DisplayName("Returns mean face")
        void returnsMeanFace() {
            double[] meanFace = extractor.getMeanFace();
            assertNotNull(meanFace);
            assertEquals(IMAGE_WIDTH * IMAGE_HEIGHT, meanFace.length);
        }

        @Test
        @DisplayName("Returns eigenfaces")
        void returnsEigenfaces() {
            double[] eigenface = extractor.getEigenface(0);
            assertNotNull(eigenface);
            assertEquals(IMAGE_WIDTH * IMAGE_HEIGHT, eigenface.length);
        }

        @Test
        @DisplayName("Returns all eigenfaces")
        void returnsAllEigenfaces() {
            double[][] eigenfaces = extractor.getAllEigenfaces();
            assertNotNull(eigenfaces);
            assertTrue(eigenfaces.length <= NUM_COMPONENTS);
        }

        @Test
        @DisplayName("Returns explained variance")
        void returnsExplainedVariance() {
            double[] variance = extractor.getExplainedVarianceRatio();
            assertNotNull(variance);

            double total = 0;
            for (double v : variance) {
                assertTrue(v >= 0 && v <= 1);
                total += v;
            }
            assertTrue(total > 0 && total <= 1.0001);
        }

        @Test
        @DisplayName("Throws on invalid eigenface index")
        void throwsOnInvalidIndex() {
            assertThrows(IndexOutOfBoundsException.class,
                () -> extractor.getEigenface(100));
        }
    }

    @Nested
    @DisplayName("Reconstruction")
    class ReconstructionTests {

        @BeforeEach
        void trainExtractor() {
            extractor.train(trainingImages, null);
        }

        @Test
        @DisplayName("Reconstructs face from features")
        void reconstructsFace() {
            FaceImage original = trainingImages.get(0);
            FeatureVector features = extractor.extract(original);
            double[] reconstructed = extractor.reconstruct(features);

            assertNotNull(reconstructed);
            assertEquals(IMAGE_WIDTH * IMAGE_HEIGHT, reconstructed.length);
        }

        @Test
        @DisplayName("Reconstruction has valid pixel values")
        void reconstructionHasValidPixels() {
            FeatureVector features = extractor.extract(trainingImages.get(0));
            double[] reconstructed = extractor.reconstruct(features);

            // Check that most values are in valid range
            int validCount = 0;
            for (double pixel : reconstructed) {
                if (pixel >= -50 && pixel <= 305) { // Allow some deviation
                    validCount++;
                }
            }
            assertTrue(validCount > reconstructed.length * 0.9);
        }
    }

    @Nested
    @DisplayName("Configuration")
    class ConfigurationTests {

        @Test
        @DisplayName("Returns correct algorithm name")
        void correctAlgorithmName() {
            assertEquals("Eigenfaces", extractor.getAlgorithmName());
        }

        @Test
        @DisplayName("Returns correct version")
        void correctVersion() {
            assertEquals(2, extractor.getVersion());
        }

        @Test
        @DisplayName("Returns expected image size")
        void correctImageSize() {
            int[] size = extractor.getExpectedImageSize();
            assertEquals(IMAGE_WIDTH, size[0]);
            assertEquals(IMAGE_HEIGHT, size[1]);
        }

        @Test
        @DisplayName("Returns config")
        void returnsConfig() {
            assertNotNull(extractor.getConfig());
            assertEquals(NUM_COMPONENTS, extractor.getConfig().getNumComponents());
        }
    }
}
