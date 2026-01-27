package com.facerecognition.domain.model;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.Objects;
import java.util.Optional;
import java.util.UUID;
import javax.imageio.ImageIO;

/**
 * Represents a face image with associated metadata.
 * This is a core domain entity that encapsulates face image data
 * along with quality metrics and processing status.
 *
 * <p>FaceImage is immutable after creation and provides methods for
 * accessing image data and metadata. Quality metrics can be computed
 * lazily when requested.</p>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * FaceImage face = FaceImage.fromFile(new File("face.jpg"));
 * System.out.println("Image size: " + face.getWidth() + "x" + face.getHeight());
 * System.out.println("Quality score: " + face.getQualityScore());
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see FaceRegion
 * @see DetectedFace
 */
public class FaceImage implements Serializable {

    private static final long serialVersionUID = 2L;

    /** Minimum acceptable image dimension in pixels. */
    public static final int MIN_DIMENSION = 20;

    /** Recommended dimension for face recognition. */
    public static final int RECOMMENDED_DIMENSION = 160;

    /** Maximum supported image dimension in pixels. */
    public static final int MAX_DIMENSION = 4096;

    private final String id;
    private final transient BufferedImage image;
    private final int width;
    private final int height;
    private final String sourcePath;
    private final LocalDateTime capturedAt;
    private final ImageFormat format;

    // Lazy-loaded quality metrics
    private transient Double qualityScore;
    private transient Double brightness;
    private transient Double contrast;
    private transient Double sharpness;

    /**
     * Supported image formats for face recognition.
     */
    public enum ImageFormat {
        /** JPEG format - commonly used for photos. */
        JPEG("jpg", "image/jpeg"),
        /** PNG format - lossless compression. */
        PNG("png", "image/png"),
        /** BMP format - uncompressed bitmap. */
        BMP("bmp", "image/bmp"),
        /** Unknown or unsupported format. */
        UNKNOWN("", "application/octet-stream");

        private final String extension;
        private final String mimeType;

        ImageFormat(String extension, String mimeType) {
            this.extension = extension;
            this.mimeType = mimeType;
        }

        public String getExtension() {
            return extension;
        }

        public String getMimeType() {
            return mimeType;
        }

        /**
         * Determines image format from file extension.
         *
         * @param filename the filename to analyze
         * @return the detected ImageFormat
         */
        public static ImageFormat fromFilename(String filename) {
            if (filename == null) return UNKNOWN;
            String lower = filename.toLowerCase();
            if (lower.endsWith(".jpg") || lower.endsWith(".jpeg")) return JPEG;
            if (lower.endsWith(".png")) return PNG;
            if (lower.endsWith(".bmp")) return BMP;
            return UNKNOWN;
        }
    }

    /**
     * Creates a new FaceImage with the specified parameters.
     *
     * @param image the BufferedImage containing the face
     * @param sourcePath the original file path (may be null)
     * @param format the image format
     * @throws IllegalArgumentException if image is null or has invalid dimensions
     */
    public FaceImage(BufferedImage image, String sourcePath, ImageFormat format) {
        Objects.requireNonNull(image, "Image cannot be null");
        validateDimensions(image.getWidth(), image.getHeight());

        this.id = UUID.randomUUID().toString();
        this.image = image;
        this.width = image.getWidth();
        this.height = image.getHeight();
        this.sourcePath = sourcePath;
        this.capturedAt = LocalDateTime.now();
        this.format = format != null ? format : ImageFormat.UNKNOWN;
    }

    /**
     * Creates a FaceImage from a file.
     *
     * @param file the image file
     * @return a new FaceImage instance
     * @throws IOException if the file cannot be read
     * @throws IllegalArgumentException if the file contains invalid image data
     */
    public static FaceImage fromFile(File file) throws IOException {
        Objects.requireNonNull(file, "File cannot be null");
        if (!file.exists()) {
            throw new IOException("File does not exist: " + file.getAbsolutePath());
        }

        BufferedImage image = ImageIO.read(file);
        if (image == null) {
            throw new IOException("Could not read image from file: " + file.getAbsolutePath());
        }

        ImageFormat format = ImageFormat.fromFilename(file.getName());
        return new FaceImage(image, file.getAbsolutePath(), format);
    }

    /**
     * Creates a FaceImage from a BufferedImage.
     *
     * @param image the BufferedImage
     * @return a new FaceImage instance
     */
    public static FaceImage fromBufferedImage(BufferedImage image) {
        return new FaceImage(image, null, ImageFormat.UNKNOWN);
    }

    private void validateDimensions(int width, int height) {
        if (width < MIN_DIMENSION || height < MIN_DIMENSION) {
            throw new IllegalArgumentException(
                String.format("Image dimensions too small: %dx%d (minimum: %dx%d)",
                    width, height, MIN_DIMENSION, MIN_DIMENSION));
        }
        if (width > MAX_DIMENSION || height > MAX_DIMENSION) {
            throw new IllegalArgumentException(
                String.format("Image dimensions too large: %dx%d (maximum: %dx%d)",
                    width, height, MAX_DIMENSION, MAX_DIMENSION));
        }
    }

    /**
     * Gets the unique identifier for this face image.
     *
     * @return the UUID string
     */
    public String getId() {
        return id;
    }

    /**
     * Gets the underlying BufferedImage.
     *
     * @return the BufferedImage
     */
    public BufferedImage getImage() {
        return image;
    }

    /**
     * Gets the image width in pixels.
     *
     * @return the width
     */
    public int getWidth() {
        return width;
    }

    /**
     * Gets the image height in pixels.
     *
     * @return the height
     */
    public int getHeight() {
        return height;
    }

    /**
     * Gets the original source file path, if available.
     *
     * @return Optional containing the path, or empty if not from a file
     */
    public Optional<String> getSourcePath() {
        return Optional.ofNullable(sourcePath);
    }

    /**
     * Gets the timestamp when this image was captured/loaded.
     *
     * @return the capture timestamp
     */
    public LocalDateTime getCapturedAt() {
        return capturedAt;
    }

    /**
     * Gets the image format.
     *
     * @return the ImageFormat
     */
    public ImageFormat getFormat() {
        return format;
    }

    /**
     * Computes and returns an overall quality score for the image.
     * The score is a value between 0.0 (poor) and 1.0 (excellent).
     *
     * <p>Quality is computed based on brightness, contrast, sharpness,
     * and resolution relative to the recommended size.</p>
     *
     * @return the quality score between 0.0 and 1.0
     */
    public double getQualityScore() {
        if (qualityScore == null) {
            computeQualityMetrics();
        }
        return qualityScore;
    }

    /**
     * Gets the average brightness of the image.
     *
     * @return brightness value between 0.0 and 1.0
     */
    public double getBrightness() {
        if (brightness == null) {
            computeQualityMetrics();
        }
        return brightness;
    }

    /**
     * Gets the contrast measure of the image.
     *
     * @return contrast value between 0.0 and 1.0
     */
    public double getContrast() {
        if (contrast == null) {
            computeQualityMetrics();
        }
        return contrast;
    }

    /**
     * Gets the sharpness measure of the image.
     *
     * @return sharpness value between 0.0 and 1.0
     */
    public double getSharpness() {
        if (sharpness == null) {
            computeQualityMetrics();
        }
        return sharpness;
    }

    /**
     * Checks if the image meets minimum quality requirements.
     *
     * @param threshold the minimum quality score required
     * @return true if quality score meets or exceeds threshold
     */
    public boolean meetsQualityThreshold(double threshold) {
        return getQualityScore() >= threshold;
    }

    private void computeQualityMetrics() {
        // Compute brightness (average pixel intensity)
        double totalBrightness = 0;
        double totalVariance = 0;
        int pixelCount = width * height;

        double[] intensities = new double[pixelCount];
        int index = 0;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                double intensity = (r + g + b) / (3.0 * 255.0);
                intensities[index++] = intensity;
                totalBrightness += intensity;
            }
        }

        brightness = totalBrightness / pixelCount;

        // Compute contrast (standard deviation of intensities)
        for (double intensity : intensities) {
            totalVariance += Math.pow(intensity - brightness, 2);
        }
        contrast = Math.sqrt(totalVariance / pixelCount);

        // Compute sharpness using Laplacian variance
        sharpness = computeLaplacianVariance();

        // Compute overall quality score
        double brightnessScore = 1.0 - Math.abs(brightness - 0.5) * 2; // Optimal at 0.5
        double contrastScore = Math.min(contrast * 4, 1.0); // Higher contrast is better
        double sharpnessScore = Math.min(sharpness * 10, 1.0); // Higher sharpness is better
        double resolutionScore = Math.min((double) Math.min(width, height) / RECOMMENDED_DIMENSION, 1.0);

        qualityScore = (brightnessScore * 0.2 + contrastScore * 0.3 +
                       sharpnessScore * 0.3 + resolutionScore * 0.2);
    }

    private double computeLaplacianVariance() {
        // Simplified Laplacian variance for sharpness estimation
        double variance = 0;
        int count = 0;

        for (int y = 1; y < height - 1; y++) {
            for (int x = 1; x < width - 1; x++) {
                int center = getGrayscale(x, y);
                int laplacian = 4 * center
                    - getGrayscale(x - 1, y)
                    - getGrayscale(x + 1, y)
                    - getGrayscale(x, y - 1)
                    - getGrayscale(x, y + 1);
                variance += laplacian * laplacian;
                count++;
            }
        }

        return count > 0 ? Math.sqrt(variance / count) / 255.0 : 0;
    }

    private int getGrayscale(int x, int y) {
        int rgb = image.getRGB(x, y);
        int r = (rgb >> 16) & 0xFF;
        int g = (rgb >> 8) & 0xFF;
        int b = rgb & 0xFF;
        return (r + g + b) / 3;
    }

    /**
     * Creates a resized copy of this face image.
     *
     * @param newWidth the target width
     * @param newHeight the target height
     * @return a new FaceImage with the specified dimensions
     */
    public FaceImage resize(int newWidth, int newHeight) {
        validateDimensions(newWidth, newHeight);

        BufferedImage resized = new BufferedImage(newWidth, newHeight, BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g = resized.createGraphics();
        g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION,
                          java.awt.RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g.drawImage(image, 0, 0, newWidth, newHeight, null);
        g.dispose();

        return new FaceImage(resized, sourcePath, format);
    }

    /**
     * Extracts pixel data as a grayscale array.
     *
     * @return array of grayscale values (0-255)
     */
    public double[] toGrayscaleArray() {
        double[] pixels = new double[width * height];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                pixels[y * width + x] = getGrayscale(x, y);
            }
        }
        return pixels;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        FaceImage faceImage = (FaceImage) o;
        return id.equals(faceImage.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return String.format("FaceImage{id='%s', size=%dx%d, format=%s, quality=%.2f}",
            id, width, height, format, getQualityScore());
    }
}
