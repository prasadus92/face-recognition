package com.facerecognition.infrastructure.extraction;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.service.FeatureExtractor;

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive unit tests for LBPHExtractor.
 */
@DisplayName("LBPHExtractor Tests")
class LBPHExtractorTest {

    private LBPHExtractor extractor;

    @BeforeEach
    void setUp() {
        extractor = new LBPHExtractor();
    }

    @Nested
    @DisplayName("LBP Computation Tests")
    class LBPComputationTests {

        @Test
        @DisplayName("Should compute LBP for uniform image")
        void shouldComputeLBPForUniformImage() {
            FaceImage uniformImage = createUniformImage(48, 64, 128);

            FeatureVector features = extractor.extract(uniformImage);

            assertThat(features).isNotNull();
            assertThat(features.getDimension()).isEqualTo(extractor.getFeatureDimension());
        }

        @Test
        @DisplayName("Should produce different LBP for different textures")
        void shouldProduceDifferentLBPForDifferentTextures() {
            FaceImage uniformImage = createUniformImage(48, 64, 128);
            FaceImage checkeredImage = createCheckeredImage(48, 64);

            FeatureVector uniformFeatures = extractor.extract(uniformImage);
            FeatureVector checkeredFeatures = extractor.extract(checkeredImage);

            // Features should be different for different textures
            double distance = uniformFeatures.euclideanDistance(checkeredFeatures);
            assertThat(distance).isGreaterThan(0.0);
        }

        @Test
        @DisplayName("Should handle edge pixels correctly")
        void shouldHandleEdgePixelsCorrectly() {
            // Create a small image to test edge handling
            FaceImage smallImage = createUniformImage(48, 64, 100);

            FeatureVector features = extractor.extract(smallImage);

            assertThat(features).isNotNull();
            assertThat(features.getFeatures()).doesNotContain(Double.NaN);
            assertThat(features.getFeatures()).doesNotContain(Double.POSITIVE_INFINITY);
        }

        @Test
        @DisplayName("Should compute similar features for similar images")
        void shouldComputeSimilarFeaturesForSimilarImages() {
            FaceImage image1 = createUniformImage(48, 64, 100);
            FaceImage image2 = createUniformImage(48, 64, 102); // Slightly different brightness

            FeatureVector features1 = extractor.extract(image1);
            FeatureVector features2 = extractor.extract(image2);

            double distance = features1.euclideanDistance(features2);
            // Similar images should have small distance
            assertThat(distance).isLessThan(1.0);
        }

        @Test
        @DisplayName("Should be invariant to uniform illumination change")
        void shouldBeInvariantToUniformIlluminationChange() {
            // Create a gradient image and a brighter version
            FaceImage image1 = createGradientImage(48, 64);
            FaceImage image2 = createBrighterGradientImage(48, 64);

            FeatureVector features1 = extractor.extract(image1);
            FeatureVector features2 = extractor.extract(image2);

            // LBP should be somewhat invariant to illumination changes
            double distance = features1.euclideanDistance(features2);
            // The distance should be relatively small
            assertThat(distance).isLessThan(10.0);
        }
    }

    @Nested
    @DisplayName("Histogram Generation Tests")
    class HistogramGenerationTests {

        @Test
        @DisplayName("Should generate histogram with correct number of bins")
        void shouldGenerateHistogramWithCorrectNumberOfBins() {
            FaceImage image = createUniformImage(48, 64, 128);

            FeatureVector features = extractor.extract(image);

            // Default: 8x8 grid, 256 bins per region
            int expectedDimension = 8 * 8 * 256;
            assertThat(features.getDimension()).isEqualTo(expectedDimension);
        }

        @Test
        @DisplayName("Should normalize histogram when configured")
        void shouldNormalizeHistogramWhenConfigured() {
            // Default extractor has normalization enabled
            FaceImage image = createUniformImage(48, 64, 128);

            FeatureVector features = extractor.extract(image);

            // Check that each region's histogram sums to approximately 1
            double[] featureArray = features.getFeatures();
            int binsPerRegion = 256;
            int numRegions = featureArray.length / binsPerRegion;

            for (int region = 0; region < numRegions; region++) {
                double sum = 0;
                for (int i = 0; i < binsPerRegion; i++) {
                    sum += featureArray[region * binsPerRegion + i];
                }
                // Normalized histograms should sum to approximately 1
                if (sum > 0) {
                    assertThat(sum).isCloseTo(1.0, within(0.01));
                }
            }
        }

        @Test
        @DisplayName("Should produce non-negative histogram values")
        void shouldProduceNonNegativeHistogramValues() {
            FaceImage image = createCheckeredImage(48, 64);

            FeatureVector features = extractor.extract(image);

            assertThat(features.getFeatures()).allMatch(v -> v >= 0);
        }
    }

    @Nested
    @DisplayName("Grid Configuration Tests")
    class GridConfigurationTests {

        @ParameterizedTest
        @DisplayName("Should create extractor with various grid sizes")
        @CsvSource({
            "4, 4",
            "8, 8",
            "10, 10",
            "8, 16"
        })
        void shouldCreateExtractorWithVariousGridSizes(int gridX, int gridY) {
            LBPHExtractor customExtractor = new LBPHExtractor(gridX, gridY, 1, 8);

            assertThat(customExtractor.getGridSize()).containsExactly(gridX, gridY);
            assertThat(customExtractor.getFeatureDimension()).isEqualTo(gridX * gridY * 256);
        }

        @Test
        @DisplayName("Should throw exception for invalid grid dimensions")
        void shouldThrowExceptionForInvalidGridDimensions() {
            assertThatIllegalArgumentException()
                .isThrownBy(() -> new LBPHExtractor(0, 8, 1, 8))
                .withMessageContaining("Grid dimensions must be positive");

            assertThatIllegalArgumentException()
                .isThrownBy(() -> new LBPHExtractor(8, 0, 1, 8))
                .withMessageContaining("Grid dimensions must be positive");

            assertThatIllegalArgumentException()
                .isThrownBy(() -> new LBPHExtractor(-1, 8, 1, 8))
                .withMessageContaining("Grid dimensions must be positive");
        }

        @Test
        @DisplayName("Should throw exception for invalid radius")
        void shouldThrowExceptionForInvalidRadius() {
            assertThatIllegalArgumentException()
                .isThrownBy(() -> new LBPHExtractor(8, 8, 0, 8))
                .withMessageContaining("Radius must be positive");

            assertThatIllegalArgumentException()
                .isThrownBy(() -> new LBPHExtractor(8, 8, -1, 8))
                .withMessageContaining("Radius must be positive");
        }

        @Test
        @DisplayName("Should throw exception for unsupported neighbor count")
        void shouldThrowExceptionForUnsupportedNeighborCount() {
            assertThatIllegalArgumentException()
                .isThrownBy(() -> new LBPHExtractor(8, 8, 1, 4))
                .withMessageContaining("Currently only 8 neighbors supported");

            assertThatIllegalArgumentException()
                .isThrownBy(() -> new LBPHExtractor(8, 8, 1, 16))
                .withMessageContaining("Currently only 8 neighbors supported");
        }

        @Test
        @DisplayName("Should expose grid parameters correctly")
        void shouldExposeGridParametersCorrectly() {
            LBPHExtractor customExtractor = new LBPHExtractor(4, 6, 2, 8);

            assertThat(customExtractor.getGridSize()).containsExactly(4, 6);
            assertThat(customExtractor.getRadius()).isEqualTo(2);
            assertThat(customExtractor.getNeighbors()).isEqualTo(8);
        }

        @Test
        @DisplayName("Should handle different grid aspect ratios")
        void shouldHandleDifferentGridAspectRatios() {
            LBPHExtractor wideGrid = new LBPHExtractor(16, 4, 1, 8);
            LBPHExtractor tallGrid = new LBPHExtractor(4, 16, 1, 8);

            FaceImage image = createUniformImage(48, 64, 128);

            FeatureVector wideFeatures = wideGrid.extract(image);
            FeatureVector tallFeatures = tallGrid.extract(image);

            assertThat(wideFeatures.getDimension()).isEqualTo(16 * 4 * 256);
            assertThat(tallFeatures.getDimension()).isEqualTo(4 * 16 * 256);
        }
    }

    @Nested
    @DisplayName("Uniform LBP Pattern Tests")
    class UniformLBPPatternTests {

        @ParameterizedTest
        @DisplayName("Should identify uniform patterns")
        @ValueSource(ints = {0, 1, 2, 3, 7, 15, 31, 63, 127, 255, 128, 192, 224, 240, 248, 252, 254})
        void shouldIdentifyUniformPatterns(int lbpCode) {
            // These are uniform patterns (0 or 2 transitions)
            assertThat(LBPHExtractor.isUniform(lbpCode)).isTrue();
        }

        @ParameterizedTest
        @DisplayName("Should identify non-uniform patterns")
        @ValueSource(ints = {5, 10, 42, 85, 170, 105, 150, 90})
        void shouldIdentifyNonUniformPatterns(int lbpCode) {
            // These have more than 2 transitions
            assertThat(LBPHExtractor.isUniform(lbpCode)).isFalse();
        }

        @Test
        @DisplayName("Should correctly count transitions for edge cases")
        void shouldCorrectlyCountTransitionsForEdgeCases() {
            // 0b10101010 = 170 (4 transitions within + possible wrap)
            assertThat(LBPHExtractor.isUniform(0b10101010)).isFalse();

            // 0b00001111 = 15 (1 transition at bit 3-4, 1 transition at wrap)
            assertThat(LBPHExtractor.isUniform(0b00001111)).isTrue();

            // 0b00000001 = 1 (transitions at bit 0-1 and bit 7-0 wrap)
            assertThat(LBPHExtractor.isUniform(0b00000001)).isTrue();
        }
    }

    @Nested
    @DisplayName("Training Tests")
    class TrainingTests {

        @Test
        @DisplayName("Should be initialized without training (LBPH does not require training)")
        void shouldBeInitializedWithoutTraining() {
            assertThat(extractor.isTrained()).isTrue();
        }

        @Test
        @DisplayName("Should accept train call without error")
        void shouldAcceptTrainCallWithoutError() {
            List<FaceImage> faces = Collections.singletonList(createUniformImage(48, 64, 128));
            List<String> labels = Collections.singletonList("person1");

            assertThatNoException().isThrownBy(() -> extractor.train(faces, labels));
            assertThat(extractor.isTrained()).isTrue();
        }

        @Test
        @DisplayName("Should throw exception for empty face list in training")
        void shouldThrowExceptionForEmptyFaceListInTraining() {
            List<FaceImage> emptyFaces = Collections.emptyList();
            List<String> labels = Collections.emptyList();

            assertThatIllegalArgumentException()
                .isThrownBy(() -> extractor.train(emptyFaces, labels))
                .withMessageContaining("Face list cannot be empty");
        }

        @Test
        @DisplayName("Should throw exception for null face list in training")
        void shouldThrowExceptionForNullFaceListInTraining() {
            assertThatIllegalArgumentException()
                .isThrownBy(() -> extractor.train(null, Collections.singletonList("label")))
                .withMessageContaining("Face list cannot be empty");
        }

        @Test
        @DisplayName("Should remain initialized after reset")
        void shouldRemainInitializedAfterReset() {
            extractor.reset();

            assertThat(extractor.isTrained()).isTrue();
        }
    }

    @Nested
    @DisplayName("Extraction Tests")
    class ExtractionTests {

        @Test
        @DisplayName("Should extract features from valid image")
        void shouldExtractFeaturesFromValidImage() {
            FaceImage image = createUniformImage(48, 64, 128);

            FeatureVector features = extractor.extract(image);

            assertThat(features).isNotNull();
            assertThat(features.getAlgorithmName()).isEqualTo("LBPH");
            assertThat(features.getAlgorithmVersion()).isEqualTo(2);
        }

        @Test
        @DisplayName("Should auto-resize image if dimensions differ")
        void shouldAutoResizeImageIfDimensionsDiffer() {
            // Create image with different dimensions
            FaceImage largeImage = createUniformImage(100, 120, 128);

            FeatureVector features = extractor.extract(largeImage);

            // Should still extract features correctly
            assertThat(features).isNotNull();
            assertThat(features.getDimension()).isEqualTo(extractor.getFeatureDimension());
        }

        @Test
        @DisplayName("Should extract batch of images")
        void shouldExtractBatchOfImages() {
            List<FaceImage> images = Arrays.asList(
                createUniformImage(48, 64, 100),
                createUniformImage(48, 64, 150),
                createUniformImage(48, 64, 200)
            );

            List<FeatureVector> features = extractor.extractBatch(images);

            assertThat(features).hasSize(3);
            assertThat(features).allMatch(f -> f.getDimension() == extractor.getFeatureDimension());
        }

        @Test
        @DisplayName("Should produce reproducible features")
        void shouldProduceReproducibleFeatures() {
            FaceImage image = createUniformImage(48, 64, 128);

            FeatureVector features1 = extractor.extract(image);
            FeatureVector features2 = extractor.extract(image);

            assertThat(features1.getFeatures()).containsExactly(features2.getFeatures());
        }
    }

    @Nested
    @DisplayName("Configuration Tests")
    class ConfigurationTests {

        @Test
        @DisplayName("Should return correct algorithm name")
        void shouldReturnCorrectAlgorithmName() {
            assertThat(extractor.getAlgorithmName()).isEqualTo("LBPH");
        }

        @Test
        @DisplayName("Should return correct version")
        void shouldReturnCorrectVersion() {
            assertThat(extractor.getVersion()).isEqualTo(2);
        }

        @Test
        @DisplayName("Should return expected image size")
        void shouldReturnExpectedImageSize() {
            int[] size = extractor.getExpectedImageSize();

            assertThat(size).hasSize(2);
            assertThat(size[0]).isEqualTo(48);  // width
            assertThat(size[1]).isEqualTo(64);  // height
        }

        @Test
        @DisplayName("Should return configuration object")
        void shouldReturnConfigurationObject() {
            FeatureExtractor.ExtractorConfig config = extractor.getConfig();

            assertThat(config).isNotNull();
            assertThat(config.isNormalize()).isTrue();
        }

        @Test
        @DisplayName("Should format toString correctly")
        void shouldFormatToStringCorrectly() {
            String str = extractor.toString();

            assertThat(str).contains("LBPHExtractor");
            assertThat(str).contains("grid=8x8");
            assertThat(str).contains("radius=1");
            assertThat(str).contains("neighbors=8");
        }
    }

    @Nested
    @DisplayName("Feature Dimension Tests")
    class FeatureDimensionTests {

        @Test
        @DisplayName("Should calculate correct feature dimension for default extractor")
        void shouldCalculateCorrectFeatureDimensionForDefault() {
            // Default: 8x8 grid, 256 bins
            assertThat(extractor.getFeatureDimension()).isEqualTo(8 * 8 * 256);
        }

        @ParameterizedTest
        @DisplayName("Should calculate correct feature dimension for custom grids")
        @CsvSource({
            "4, 4, 4096",
            "8, 8, 16384",
            "10, 12, 30720",
            "2, 2, 1024"
        })
        void shouldCalculateCorrectFeatureDimensionForCustomGrids(int gridX, int gridY, int expectedDim) {
            LBPHExtractor customExtractor = new LBPHExtractor(gridX, gridY, 1, 8);

            assertThat(customExtractor.getFeatureDimension()).isEqualTo(expectedDim);
        }
    }

    // Helper methods

    private FaceImage createUniformImage(int width, int height, int grayValue) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = image.createGraphics();
        g.setColor(new Color(grayValue, grayValue, grayValue));
        g.fillRect(0, 0, width, height);
        g.dispose();
        return FaceImage.fromBufferedImage(image);
    }

    private FaceImage createCheckeredImage(int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int color = ((x + y) % 2 == 0) ? 255 : 0;
                image.setRGB(x, y, new Color(color, color, color).getRGB());
            }
        }
        return FaceImage.fromBufferedImage(image);
    }

    private FaceImage createGradientImage(int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = (int) ((x / (double) width) * 255);
                image.setRGB(x, y, new Color(gray, gray, gray).getRGB());
            }
        }
        return FaceImage.fromBufferedImage(image);
    }

    private FaceImage createBrighterGradientImage(int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = Math.min(255, (int) ((x / (double) width) * 255) + 50);
                image.setRGB(x, y, new Color(gray, gray, gray).getRGB());
            }
        }
        return FaceImage.fromBufferedImage(image);
    }
}
