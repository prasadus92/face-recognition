package com.facerecognition.domain.model;

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.time.LocalDateTime;
import javax.imageio.ImageIO;

import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive unit tests for FaceImage domain model.
 */
@DisplayName("FaceImage Tests")
class FaceImageTest {

    private static final int TEST_WIDTH = 100;
    private static final int TEST_HEIGHT = 100;

    @Nested
    @DisplayName("Creation from BufferedImage")
    class CreationFromBufferedImage {

        @Test
        @DisplayName("Should create FaceImage from valid BufferedImage")
        void shouldCreateFromValidBufferedImage() {
            BufferedImage bufferedImage = createTestImage(TEST_WIDTH, TEST_HEIGHT, Color.WHITE);

            FaceImage faceImage = FaceImage.fromBufferedImage(bufferedImage);

            assertThat(faceImage).isNotNull();
            assertThat(faceImage.getWidth()).isEqualTo(TEST_WIDTH);
            assertThat(faceImage.getHeight()).isEqualTo(TEST_HEIGHT);
            assertThat(faceImage.getId()).isNotNull().isNotEmpty();
            assertThat(faceImage.getCapturedAt()).isBeforeOrEqualTo(LocalDateTime.now());
            assertThat(faceImage.getFormat()).isEqualTo(FaceImage.ImageFormat.UNKNOWN);
            assertThat(faceImage.getSourcePath()).isEmpty();
        }

        @Test
        @DisplayName("Should throw exception for null BufferedImage")
        void shouldThrowExceptionForNullBufferedImage() {
            assertThatNullPointerException()
                .isThrownBy(() -> FaceImage.fromBufferedImage(null))
                .withMessage("Image cannot be null");
        }

        @Test
        @DisplayName("Should create FaceImage with TYPE_INT_RGB format")
        void shouldCreateFromTypeIntRgb() {
            BufferedImage bufferedImage = new BufferedImage(50, 50, BufferedImage.TYPE_INT_RGB);

            FaceImage faceImage = FaceImage.fromBufferedImage(bufferedImage);

            assertThat(faceImage).isNotNull();
            assertThat(faceImage.getImage()).isNotNull();
        }

        @Test
        @DisplayName("Should create FaceImage with TYPE_INT_ARGB format")
        void shouldCreateFromTypeIntArgb() {
            BufferedImage bufferedImage = new BufferedImage(50, 50, BufferedImage.TYPE_INT_ARGB);

            FaceImage faceImage = FaceImage.fromBufferedImage(bufferedImage);

            assertThat(faceImage).isNotNull();
        }

        @Test
        @DisplayName("Should create FaceImage with TYPE_BYTE_GRAY format")
        void shouldCreateFromTypeByteGray() {
            BufferedImage bufferedImage = new BufferedImage(50, 50, BufferedImage.TYPE_BYTE_GRAY);

            FaceImage faceImage = FaceImage.fromBufferedImage(bufferedImage);

            assertThat(faceImage).isNotNull();
        }
    }

    @Nested
    @DisplayName("Quality Metrics Calculation")
    class QualityMetricsCalculation {

        @Test
        @DisplayName("Should calculate quality score for uniform gray image")
        void shouldCalculateQualityScoreForUniformGrayImage() {
            BufferedImage grayImage = createTestImage(100, 100, new Color(128, 128, 128));

            FaceImage faceImage = FaceImage.fromBufferedImage(grayImage);

            double qualityScore = faceImage.getQualityScore();

            assertThat(qualityScore).isBetween(0.0, 1.0);
        }

        @Test
        @DisplayName("Should calculate brightness for white image")
        void shouldCalculateBrightnessForWhiteImage() {
            BufferedImage whiteImage = createTestImage(100, 100, Color.WHITE);

            FaceImage faceImage = FaceImage.fromBufferedImage(whiteImage);

            double brightness = faceImage.getBrightness();

            assertThat(brightness).isCloseTo(1.0, within(0.01));
        }

        @Test
        @DisplayName("Should calculate brightness for black image")
        void shouldCalculateBrightnessForBlackImage() {
            BufferedImage blackImage = createTestImage(100, 100, Color.BLACK);

            FaceImage faceImage = FaceImage.fromBufferedImage(blackImage);

            double brightness = faceImage.getBrightness();

            assertThat(brightness).isCloseTo(0.0, within(0.01));
        }

        @Test
        @DisplayName("Should calculate contrast for uniform image")
        void shouldCalculateContrastForUniformImage() {
            BufferedImage uniformImage = createTestImage(100, 100, Color.GRAY);

            FaceImage faceImage = FaceImage.fromBufferedImage(uniformImage);

            double contrast = faceImage.getContrast();

            assertThat(contrast).isCloseTo(0.0, within(0.01));
        }

        @Test
        @DisplayName("Should calculate higher contrast for checkered image")
        void shouldCalculateHigherContrastForCheckeredImage() {
            BufferedImage checkeredImage = createCheckeredImage(100, 100);

            FaceImage faceImage = FaceImage.fromBufferedImage(checkeredImage);

            double contrast = faceImage.getContrast();

            assertThat(contrast).isGreaterThan(0.0);
        }

        @Test
        @DisplayName("Should calculate sharpness")
        void shouldCalculateSharpness() {
            BufferedImage image = createTestImage(100, 100, Color.GRAY);

            FaceImage faceImage = FaceImage.fromBufferedImage(image);

            double sharpness = faceImage.getSharpness();

            assertThat(sharpness).isGreaterThanOrEqualTo(0.0);
        }

        @Test
        @DisplayName("Should meet quality threshold when quality is sufficient")
        void shouldMeetQualityThreshold() {
            BufferedImage image = createTestImage(160, 160, Color.GRAY);

            FaceImage faceImage = FaceImage.fromBufferedImage(image);

            assertThat(faceImage.meetsQualityThreshold(0.0)).isTrue();
        }

        @Test
        @DisplayName("Should not meet high quality threshold for simple image")
        void shouldNotMeetHighQualityThreshold() {
            BufferedImage image = createTestImage(30, 30, Color.BLACK);

            FaceImage faceImage = FaceImage.fromBufferedImage(image);

            assertThat(faceImage.meetsQualityThreshold(0.95)).isFalse();
        }

        @Test
        @DisplayName("Quality metrics should be cached")
        void qualityMetricsShouldBeCached() {
            BufferedImage image = createTestImage(100, 100, Color.GRAY);
            FaceImage faceImage = FaceImage.fromBufferedImage(image);

            double firstCall = faceImage.getQualityScore();
            double secondCall = faceImage.getQualityScore();

            assertThat(firstCall).isEqualTo(secondCall);
        }
    }

    @Nested
    @DisplayName("Resize Functionality")
    class ResizeFunctionality {

        @Test
        @DisplayName("Should resize image to specified dimensions")
        void shouldResizeToSpecifiedDimensions() {
            BufferedImage original = createTestImage(200, 200, Color.RED);
            FaceImage faceImage = FaceImage.fromBufferedImage(original);

            FaceImage resized = faceImage.resize(100, 100);

            assertThat(resized.getWidth()).isEqualTo(100);
            assertThat(resized.getHeight()).isEqualTo(100);
        }

        @Test
        @DisplayName("Should preserve source path after resize")
        void shouldPreserveSourcePathAfterResize() throws IOException {
            File tempFile = File.createTempFile("test", ".png");
            tempFile.deleteOnExit();
            BufferedImage original = createTestImage(100, 100, Color.BLUE);
            ImageIO.write(original, "png", tempFile);

            FaceImage faceImage = FaceImage.fromFile(tempFile);
            FaceImage resized = faceImage.resize(50, 50);

            assertThat(resized.getSourcePath()).isEqualTo(faceImage.getSourcePath());
        }

        @Test
        @DisplayName("Should upscale image")
        void shouldUpscaleImage() {
            BufferedImage original = createTestImage(50, 50, Color.GREEN);
            FaceImage faceImage = FaceImage.fromBufferedImage(original);

            FaceImage resized = faceImage.resize(150, 150);

            assertThat(resized.getWidth()).isEqualTo(150);
            assertThat(resized.getHeight()).isEqualTo(150);
        }

        @Test
        @DisplayName("Should downscale image")
        void shouldDownscaleImage() {
            BufferedImage original = createTestImage(200, 200, Color.YELLOW);
            FaceImage faceImage = FaceImage.fromBufferedImage(original);

            FaceImage resized = faceImage.resize(50, 50);

            assertThat(resized.getWidth()).isEqualTo(50);
            assertThat(resized.getHeight()).isEqualTo(50);
        }

        @ParameterizedTest
        @DisplayName("Should throw exception for dimensions below minimum")
        @CsvSource({
            "10, 50",
            "50, 10",
            "5, 5"
        })
        void shouldThrowExceptionForDimensionsBelowMinimum(int width, int height) {
            BufferedImage original = createTestImage(100, 100, Color.GRAY);
            FaceImage faceImage = FaceImage.fromBufferedImage(original);

            assertThatIllegalArgumentException()
                .isThrownBy(() -> faceImage.resize(width, height))
                .withMessageContaining("Image dimensions too small");
        }

        @ParameterizedTest
        @DisplayName("Should throw exception for dimensions above maximum")
        @CsvSource({
            "5000, 100",
            "100, 5000",
            "5000, 5000"
        })
        void shouldThrowExceptionForDimensionsAboveMaximum(int width, int height) {
            BufferedImage original = createTestImage(100, 100, Color.GRAY);
            FaceImage faceImage = FaceImage.fromBufferedImage(original);

            assertThatIllegalArgumentException()
                .isThrownBy(() -> faceImage.resize(width, height))
                .withMessageContaining("Image dimensions too large");
        }

        @Test
        @DisplayName("Should create new FaceImage instance on resize")
        void shouldCreateNewInstanceOnResize() {
            BufferedImage original = createTestImage(100, 100, Color.GRAY);
            FaceImage faceImage = FaceImage.fromBufferedImage(original);

            FaceImage resized = faceImage.resize(50, 50);

            assertThat(resized).isNotSameAs(faceImage);
            assertThat(resized.getId()).isNotEqualTo(faceImage.getId());
        }
    }

    @Nested
    @DisplayName("Grayscale Array Conversion")
    class GrayscaleArrayConversion {

        @Test
        @DisplayName("Should convert to grayscale array with correct length")
        void shouldConvertToGrayscaleArrayWithCorrectLength() {
            BufferedImage image = createTestImage(50, 40, Color.GRAY);
            FaceImage faceImage = FaceImage.fromBufferedImage(image);

            double[] grayscale = faceImage.toGrayscaleArray();

            assertThat(grayscale).hasSize(50 * 40);
        }

        @Test
        @DisplayName("Should convert white image to high grayscale values")
        void shouldConvertWhiteImageToHighValues() {
            BufferedImage whiteImage = createTestImage(50, 50, Color.WHITE);
            FaceImage faceImage = FaceImage.fromBufferedImage(whiteImage);

            double[] grayscale = faceImage.toGrayscaleArray();

            assertThat(java.util.Arrays.stream(grayscale).allMatch(v -> v == 255.0)).isTrue();
        }

        @Test
        @DisplayName("Should convert black image to zero grayscale values")
        void shouldConvertBlackImageToZeroValues() {
            BufferedImage blackImage = createTestImage(50, 50, Color.BLACK);
            FaceImage faceImage = FaceImage.fromBufferedImage(blackImage);

            double[] grayscale = faceImage.toGrayscaleArray();

            assertThat(java.util.Arrays.stream(grayscale).allMatch(v -> v == 0.0)).isTrue();
        }

        @Test
        @DisplayName("Should convert red to correct grayscale value")
        void shouldConvertRedToCorrectGrayscale() {
            BufferedImage redImage = createTestImage(50, 50, Color.RED);
            FaceImage faceImage = FaceImage.fromBufferedImage(redImage);

            double[] grayscale = faceImage.toGrayscaleArray();

            // Red (255, 0, 0) -> grayscale = (255 + 0 + 0) / 3 = 85
            assertThat(grayscale[0]).isEqualTo(85.0);
        }

        @Test
        @DisplayName("Should maintain pixel ordering row by row")
        void shouldMaintainPixelOrderingRowByRow() {
            BufferedImage image = new BufferedImage(3, 2, BufferedImage.TYPE_INT_RGB);
            // Set specific pixels
            image.setRGB(0, 0, Color.RED.getRGB());
            image.setRGB(1, 0, Color.GREEN.getRGB());
            image.setRGB(2, 0, Color.BLUE.getRGB());
            image.setRGB(0, 1, Color.WHITE.getRGB());
            image.setRGB(1, 1, Color.BLACK.getRGB());
            image.setRGB(2, 1, Color.GRAY.getRGB());

            FaceImage faceImage = FaceImage.fromBufferedImage(image);
            double[] grayscale = faceImage.toGrayscaleArray();

            assertThat(grayscale).hasSize(6);
            // First row
            assertThat(grayscale[0]).isEqualTo(85.0); // Red
            assertThat(grayscale[1]).isEqualTo(85.0); // Green
            assertThat(grayscale[2]).isEqualTo(85.0); // Blue
            // Second row
            assertThat(grayscale[3]).isEqualTo(255.0); // White
            assertThat(grayscale[4]).isEqualTo(0.0); // Black
        }
    }

    @Nested
    @DisplayName("Validation")
    class Validation {

        @Test
        @DisplayName("Should validate minimum dimension")
        void shouldValidateMinimumDimension() {
            BufferedImage tooSmall = new BufferedImage(10, 10, BufferedImage.TYPE_INT_RGB);

            assertThatIllegalArgumentException()
                .isThrownBy(() -> FaceImage.fromBufferedImage(tooSmall))
                .withMessageContaining("Image dimensions too small");
        }

        @Test
        @DisplayName("Should validate maximum dimension")
        void shouldValidateMaximumDimension() {
            BufferedImage tooLarge = new BufferedImage(5000, 5000, BufferedImage.TYPE_INT_RGB);

            assertThatIllegalArgumentException()
                .isThrownBy(() -> FaceImage.fromBufferedImage(tooLarge))
                .withMessageContaining("Image dimensions too large");
        }

        @ParameterizedTest
        @DisplayName("Should accept valid dimensions")
        @CsvSource({
            "20, 20",
            "100, 100",
            "4096, 4096",
            "160, 160"
        })
        void shouldAcceptValidDimensions(int width, int height) {
            BufferedImage validImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);

            FaceImage faceImage = FaceImage.fromBufferedImage(validImage);

            assertThat(faceImage.getWidth()).isEqualTo(width);
            assertThat(faceImage.getHeight()).isEqualTo(height);
        }

        @ParameterizedTest
        @DisplayName("Should reject invalid dimensions")
        @ValueSource(ints = {1, 5, 10, 19})
        void shouldRejectInvalidDimensions(int size) {
            BufferedImage invalidImage = new BufferedImage(size, size, BufferedImage.TYPE_INT_RGB);

            assertThatIllegalArgumentException()
                .isThrownBy(() -> FaceImage.fromBufferedImage(invalidImage));
        }
    }

    @Nested
    @DisplayName("File Operations")
    class FileOperations {

        @Test
        @DisplayName("Should create from PNG file")
        void shouldCreateFromPngFile() throws IOException {
            File tempFile = File.createTempFile("test", ".png");
            tempFile.deleteOnExit();
            BufferedImage image = createTestImage(100, 100, Color.BLUE);
            ImageIO.write(image, "png", tempFile);

            FaceImage faceImage = FaceImage.fromFile(tempFile);

            assertThat(faceImage).isNotNull();
            assertThat(faceImage.getFormat()).isEqualTo(FaceImage.ImageFormat.PNG);
            assertThat(faceImage.getSourcePath()).contains(tempFile.getAbsolutePath());
        }

        @Test
        @DisplayName("Should create from JPG file")
        void shouldCreateFromJpgFile() throws IOException {
            File tempFile = File.createTempFile("test", ".jpg");
            tempFile.deleteOnExit();
            BufferedImage image = createTestImage(100, 100, Color.RED);
            ImageIO.write(image, "jpg", tempFile);

            FaceImage faceImage = FaceImage.fromFile(tempFile);

            assertThat(faceImage).isNotNull();
            assertThat(faceImage.getFormat()).isEqualTo(FaceImage.ImageFormat.JPEG);
        }

        @Test
        @DisplayName("Should throw IOException for non-existent file")
        void shouldThrowIOExceptionForNonExistentFile() {
            File nonExistent = new File("/non/existent/file.png");

            assertThatIOException()
                .isThrownBy(() -> FaceImage.fromFile(nonExistent))
                .withMessageContaining("File does not exist");
        }

        @Test
        @DisplayName("Should throw NullPointerException for null file")
        void shouldThrowNullPointerExceptionForNullFile() {
            assertThatNullPointerException()
                .isThrownBy(() -> FaceImage.fromFile(null))
                .withMessage("File cannot be null");
        }
    }

    @Nested
    @DisplayName("ImageFormat Detection")
    class ImageFormatDetection {

        @ParameterizedTest
        @DisplayName("Should detect image format from filename")
        @CsvSource({
            "image.jpg, JPEG",
            "image.jpeg, JPEG",
            "image.JPG, JPEG",
            "image.png, PNG",
            "image.PNG, PNG",
            "image.bmp, BMP",
            "image.BMP, BMP",
            "image.txt, UNKNOWN",
            "image, UNKNOWN"
        })
        void shouldDetectImageFormatFromFilename(String filename, String expectedFormat) {
            FaceImage.ImageFormat format = FaceImage.ImageFormat.fromFilename(filename);

            assertThat(format.name()).isEqualTo(expectedFormat);
        }

        @Test
        @DisplayName("Should return UNKNOWN for null filename")
        void shouldReturnUnknownForNullFilename() {
            FaceImage.ImageFormat format = FaceImage.ImageFormat.fromFilename(null);

            assertThat(format).isEqualTo(FaceImage.ImageFormat.UNKNOWN);
        }
    }

    @Nested
    @DisplayName("Equality and HashCode")
    class EqualityAndHashCode {

        @Test
        @DisplayName("Should be equal to itself")
        void shouldBeEqualToItself() {
            BufferedImage image = createTestImage(100, 100, Color.GRAY);
            FaceImage faceImage = FaceImage.fromBufferedImage(image);

            assertThat(faceImage).isEqualTo(faceImage);
        }

        @Test
        @DisplayName("Should not be equal to different instance")
        void shouldNotBeEqualToDifferentInstance() {
            BufferedImage image = createTestImage(100, 100, Color.GRAY);
            FaceImage faceImage1 = FaceImage.fromBufferedImage(image);
            FaceImage faceImage2 = FaceImage.fromBufferedImage(image);

            assertThat(faceImage1).isNotEqualTo(faceImage2);
        }

        @Test
        @DisplayName("Should not be equal to null")
        void shouldNotBeEqualToNull() {
            BufferedImage image = createTestImage(100, 100, Color.GRAY);
            FaceImage faceImage = FaceImage.fromBufferedImage(image);

            assertThat(faceImage).isNotEqualTo(null);
        }

        @Test
        @DisplayName("Should have consistent hashCode")
        void shouldHaveConsistentHashCode() {
            BufferedImage image = createTestImage(100, 100, Color.GRAY);
            FaceImage faceImage = FaceImage.fromBufferedImage(image);

            int hash1 = faceImage.hashCode();
            int hash2 = faceImage.hashCode();

            assertThat(hash1).isEqualTo(hash2);
        }
    }

    @Nested
    @DisplayName("ToString")
    class ToStringTests {

        @Test
        @DisplayName("Should include relevant information in toString")
        void shouldIncludeRelevantInformation() {
            BufferedImage image = createTestImage(100, 100, Color.GRAY);
            FaceImage faceImage = FaceImage.fromBufferedImage(image);

            String str = faceImage.toString();

            assertThat(str).contains("FaceImage");
            assertThat(str).contains("100x100");
            assertThat(str).contains("UNKNOWN");
            assertThat(str).contains("quality=");
        }
    }

    // Helper methods

    private BufferedImage createTestImage(int width, int height, Color color) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = image.createGraphics();
        g.setColor(color);
        g.fillRect(0, 0, width, height);
        g.dispose();
        return image;
    }

    private BufferedImage createCheckeredImage(int width, int height) {
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if ((x + y) % 2 == 0) {
                    image.setRGB(x, y, Color.WHITE.getRGB());
                } else {
                    image.setRGB(x, y, Color.BLACK.getRGB());
                }
            }
        }
        return image;
    }
}
