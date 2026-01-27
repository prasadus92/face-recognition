package com.facerecognition.infrastructure.preprocessing;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FaceLandmarks;
import com.facerecognition.domain.model.FaceRegion;

import java.awt.*;
import java.awt.geom.AffineTransform;
import java.awt.image.AffineTransformOp;
import java.awt.image.BufferedImage;
import java.util.Objects;

/**
 * Face alignment and normalization service.
 *
 * <p>This class provides methods to align and normalize detected faces
 * to a canonical pose, improving recognition accuracy by removing
 * variations due to head pose and position.</p>
 *
 * <h3>Alignment Process:</h3>
 * <ol>
 *   <li>Detect facial landmarks (eyes, nose, mouth)</li>
 *   <li>Calculate rotation angle from eye positions</li>
 *   <li>Apply affine transformation to align eyes horizontally</li>
 *   <li>Scale to target size</li>
 *   <li>Crop to face region</li>
 *   <li>Apply histogram equalization (optional)</li>
 * </ol>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * FaceAligner aligner = new FaceAligner.Builder()
 *     .targetSize(160, 160)
 *     .eyePositionRatio(0.35)
 *     .histogramEqualization(true)
 *     .build();
 *
 * FaceImage aligned = aligner.align(faceImage, landmarks);
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
public class FaceAligner {

    /** Default target width for aligned faces. */
    public static final int DEFAULT_TARGET_WIDTH = 160;

    /** Default target height for aligned faces. */
    public static final int DEFAULT_TARGET_HEIGHT = 160;

    /** Default ratio for eye position from top. */
    public static final double DEFAULT_EYE_POSITION_RATIO = 0.35;

    /** Default ratio for desired eye distance. */
    public static final double DEFAULT_EYE_DISTANCE_RATIO = 0.35;

    private final int targetWidth;
    private final int targetHeight;
    private final double eyePositionRatio;
    private final double eyeDistanceRatio;
    private final boolean histogramEqualization;
    private final boolean grayscale;
    private final Color backgroundColor;

    /**
     * Builder for creating FaceAligner instances.
     */
    public static class Builder {
        private int targetWidth = DEFAULT_TARGET_WIDTH;
        private int targetHeight = DEFAULT_TARGET_HEIGHT;
        private double eyePositionRatio = DEFAULT_EYE_POSITION_RATIO;
        private double eyeDistanceRatio = DEFAULT_EYE_DISTANCE_RATIO;
        private boolean histogramEqualization = false;
        private boolean grayscale = true;
        private Color backgroundColor = Color.BLACK;

        public Builder targetSize(int width, int height) {
            this.targetWidth = width;
            this.targetHeight = height;
            return this;
        }

        public Builder eyePositionRatio(double ratio) {
            this.eyePositionRatio = ratio;
            return this;
        }

        public Builder eyeDistanceRatio(double ratio) {
            this.eyeDistanceRatio = ratio;
            return this;
        }

        public Builder histogramEqualization(boolean enable) {
            this.histogramEqualization = enable;
            return this;
        }

        public Builder grayscale(boolean enable) {
            this.grayscale = enable;
            return this;
        }

        public Builder backgroundColor(Color color) {
            this.backgroundColor = color;
            return this;
        }

        public FaceAligner build() {
            return new FaceAligner(this);
        }
    }

    private FaceAligner(Builder builder) {
        this.targetWidth = builder.targetWidth;
        this.targetHeight = builder.targetHeight;
        this.eyePositionRatio = builder.eyePositionRatio;
        this.eyeDistanceRatio = builder.eyeDistanceRatio;
        this.histogramEqualization = builder.histogramEqualization;
        this.grayscale = builder.grayscale;
        this.backgroundColor = builder.backgroundColor;
    }

    /**
     * Creates a default FaceAligner.
     *
     * @return a new FaceAligner with default settings
     */
    public static FaceAligner createDefault() {
        return new Builder().build();
    }

    /**
     * Aligns a face image using detected landmarks.
     *
     * @param faceImage the input face image
     * @param landmarks the detected facial landmarks
     * @return the aligned face image
     */
    public FaceImage align(FaceImage faceImage, FaceLandmarks landmarks) {
        Objects.requireNonNull(faceImage, "Face image cannot be null");
        Objects.requireNonNull(landmarks, "Landmarks cannot be null");

        BufferedImage image = faceImage.getImage();

        // Get eye positions
        Point leftEye = landmarks.getLeftEye();
        Point rightEye = landmarks.getRightEye();

        // Calculate angle and scale
        double dx = rightEye.x - leftEye.x;
        double dy = rightEye.y - leftEye.y;
        double angle = Math.atan2(dy, dx);
        double currentEyeDistance = Math.sqrt(dx * dx + dy * dy);

        // Calculate desired eye distance based on target size
        double desiredEyeDistance = eyeDistanceRatio * targetWidth;
        double scale = desiredEyeDistance / currentEyeDistance;

        // Calculate center point between eyes
        double eyeCenterX = (leftEye.x + rightEye.x) / 2.0;
        double eyeCenterY = (leftEye.y + rightEye.y) / 2.0;

        // Calculate desired eye center position in output
        double desiredEyeCenterX = targetWidth / 2.0;
        double desiredEyeCenterY = eyePositionRatio * targetHeight;

        // Build affine transformation
        AffineTransform transform = new AffineTransform();

        // Translate to place eye center at desired position
        transform.translate(desiredEyeCenterX, desiredEyeCenterY);

        // Rotate and scale around eye center
        transform.rotate(-angle);
        transform.scale(scale, scale);

        // Translate to center on eye center
        transform.translate(-eyeCenterX, -eyeCenterY);

        // Apply transformation
        BufferedImage aligned = new BufferedImage(targetWidth, targetHeight,
            grayscale ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_INT_RGB);

        Graphics2D g2d = aligned.createGraphics();
        g2d.setColor(backgroundColor);
        g2d.fillRect(0, 0, targetWidth, targetHeight);
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                            RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING,
                            RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.drawImage(image, transform, null);
        g2d.dispose();

        // Apply histogram equalization if enabled
        if (histogramEqualization) {
            aligned = equalizeHistogram(aligned);
        }

        return FaceImage.fromBufferedImage(aligned);
    }

    /**
     * Aligns a face image using a face region (without landmarks).
     * Uses simple center-crop and resize approach.
     *
     * @param faceImage the input face image
     * @param region the detected face region
     * @return the aligned face image
     */
    public FaceImage alignFromRegion(FaceImage faceImage, FaceRegion region) {
        Objects.requireNonNull(faceImage, "Face image cannot be null");
        Objects.requireNonNull(region, "Face region cannot be null");

        BufferedImage image = faceImage.getImage();

        // Expand region slightly to include more context
        int expandX = (int) (region.getWidth() * 0.1);
        int expandY = (int) (region.getHeight() * 0.1);

        int x = Math.max(0, region.getX() - expandX);
        int y = Math.max(0, region.getY() - expandY);
        int width = Math.min(image.getWidth() - x, region.getWidth() + 2 * expandX);
        int height = Math.min(image.getHeight() - y, region.getHeight() + 2 * expandY);

        // Crop face region
        BufferedImage cropped = image.getSubimage(x, y, width, height);

        // Resize to target size
        BufferedImage resized = new BufferedImage(targetWidth, targetHeight,
            grayscale ? BufferedImage.TYPE_BYTE_GRAY : BufferedImage.TYPE_INT_RGB);

        Graphics2D g2d = resized.createGraphics();
        g2d.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                            RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        g2d.drawImage(cropped, 0, 0, targetWidth, targetHeight, null);
        g2d.dispose();

        // Apply histogram equalization if enabled
        if (histogramEqualization) {
            resized = equalizeHistogram(resized);
        }

        return FaceImage.fromBufferedImage(resized);
    }

    /**
     * Applies histogram equalization to improve contrast.
     *
     * @param image the input image
     * @return the equalized image
     */
    private BufferedImage equalizeHistogram(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int totalPixels = width * height;

        // Build histogram
        int[] histogram = new int[256];
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                int gray = (rgb >> 16) & 0xFF; // Assume grayscale or use red channel
                histogram[gray]++;
            }
        }

        // Build cumulative distribution function
        int[] cdf = new int[256];
        cdf[0] = histogram[0];
        for (int i = 1; i < 256; i++) {
            cdf[i] = cdf[i - 1] + histogram[i];
        }

        // Find minimum non-zero CDF value
        int cdfMin = 0;
        for (int i = 0; i < 256; i++) {
            if (cdf[i] > 0) {
                cdfMin = cdf[i];
                break;
            }
        }

        // Build lookup table
        int[] lut = new int[256];
        for (int i = 0; i < 256; i++) {
            lut[i] = Math.round(((float) (cdf[i] - cdfMin) / (totalPixels - cdfMin)) * 255);
            lut[i] = Math.max(0, Math.min(255, lut[i]));
        }

        // Apply equalization
        BufferedImage result = new BufferedImage(width, height, image.getType());
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                int gray = (rgb >> 16) & 0xFF;
                int newGray = lut[gray];
                int newRgb = (newGray << 16) | (newGray << 8) | newGray;
                result.setRGB(x, y, newRgb);
            }
        }

        return result;
    }

    /**
     * Applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
     *
     * @param image the input image
     * @param tileSize the size of tiles for local equalization
     * @param clipLimit the contrast limit
     * @return the equalized image
     */
    public BufferedImage applyCLAHE(BufferedImage image, int tileSize, double clipLimit) {
        int width = image.getWidth();
        int height = image.getHeight();

        int tilesX = (int) Math.ceil((double) width / tileSize);
        int tilesY = (int) Math.ceil((double) height / tileSize);

        // Build lookup tables for each tile
        int[][][] tileLUTs = new int[tilesY][tilesX][256];

        for (int ty = 0; ty < tilesY; ty++) {
            for (int tx = 0; tx < tilesX; tx++) {
                int startX = tx * tileSize;
                int startY = ty * tileSize;
                int endX = Math.min(startX + tileSize, width);
                int endY = Math.min(startY + tileSize, height);

                // Build histogram for this tile
                int[] histogram = new int[256];
                int pixelCount = 0;

                for (int y = startY; y < endY; y++) {
                    for (int x = startX; x < endX; x++) {
                        int gray = (image.getRGB(x, y) >> 16) & 0xFF;
                        histogram[gray]++;
                        pixelCount++;
                    }
                }

                // Apply clip limit
                int clipThreshold = (int) (clipLimit * pixelCount / 256);
                int excess = 0;

                for (int i = 0; i < 256; i++) {
                    if (histogram[i] > clipThreshold) {
                        excess += histogram[i] - clipThreshold;
                        histogram[i] = clipThreshold;
                    }
                }

                // Redistribute excess
                int redistribution = excess / 256;
                for (int i = 0; i < 256; i++) {
                    histogram[i] += redistribution;
                }

                // Build CDF and LUT
                int[] cdf = new int[256];
                cdf[0] = histogram[0];
                for (int i = 1; i < 256; i++) {
                    cdf[i] = cdf[i - 1] + histogram[i];
                }

                int cdfMin = 0;
                for (int i = 0; i < 256; i++) {
                    if (cdf[i] > 0) {
                        cdfMin = cdf[i];
                        break;
                    }
                }

                for (int i = 0; i < 256; i++) {
                    if (pixelCount > cdfMin) {
                        tileLUTs[ty][tx][i] = (int) Math.round(
                            ((double) (cdf[i] - cdfMin) / (pixelCount - cdfMin)) * 255);
                    } else {
                        tileLUTs[ty][tx][i] = i;
                    }
                }
            }
        }

        // Apply with bilinear interpolation between tiles
        BufferedImage result = new BufferedImage(width, height, image.getType());

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int gray = (image.getRGB(x, y) >> 16) & 0xFF;

                // Find surrounding tiles
                double tileX = (double) x / tileSize - 0.5;
                double tileY = (double) y / tileSize - 0.5;

                int tx1 = Math.max(0, (int) Math.floor(tileX));
                int tx2 = Math.min(tilesX - 1, tx1 + 1);
                int ty1 = Math.max(0, (int) Math.floor(tileY));
                int ty2 = Math.min(tilesY - 1, ty1 + 1);

                double fx = tileX - tx1;
                double fy = tileY - ty1;

                // Bilinear interpolation
                double v1 = tileLUTs[ty1][tx1][gray] * (1 - fx) + tileLUTs[ty1][tx2][gray] * fx;
                double v2 = tileLUTs[ty2][tx1][gray] * (1 - fx) + tileLUTs[ty2][tx2][gray] * fx;
                int newGray = (int) Math.round(v1 * (1 - fy) + v2 * fy);
                newGray = Math.max(0, Math.min(255, newGray));

                int newRgb = (newGray << 16) | (newGray << 8) | newGray;
                result.setRGB(x, y, newRgb);
            }
        }

        return result;
    }

    /**
     * Gets the target width.
     *
     * @return target width in pixels
     */
    public int getTargetWidth() {
        return targetWidth;
    }

    /**
     * Gets the target height.
     *
     * @return target height in pixels
     */
    public int getTargetHeight() {
        return targetHeight;
    }

    /**
     * Checks if histogram equalization is enabled.
     *
     * @return true if enabled
     */
    public boolean isHistogramEqualizationEnabled() {
        return histogramEqualization;
    }

    @Override
    public String toString() {
        return String.format("FaceAligner{target=%dx%d, histeq=%s}",
            targetWidth, targetHeight, histogramEqualization);
    }
}
