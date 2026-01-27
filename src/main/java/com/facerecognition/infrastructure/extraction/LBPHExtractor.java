package com.facerecognition.infrastructure.extraction;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.domain.service.FeatureExtractor.ExtractorConfig;

import java.io.Serializable;
import java.util.List;
import java.util.Objects;

/**
 * Local Binary Pattern Histogram (LBPH) feature extractor.
 *
 * <p>LBPH is a texture-based face recognition algorithm that captures
 * local structures in the face image. It is particularly robust to
 * illumination changes and can handle varying expressions.</p>
 *
 * <h3>Algorithm Overview:</h3>
 * <ol>
 *   <li>Divide face into grid of regions</li>
 *   <li>Compute LBP for each pixel in each region</li>
 *   <li>Build histogram of LBP values per region</li>
 *   <li>Concatenate all histograms into feature vector</li>
 * </ol>
 *
 * <h3>LBP Computation:</h3>
 * <pre>
 * For each pixel, compare with neighbors in a circular pattern.
 * If neighbor >= center: bit = 1, else bit = 0
 * Concatenate bits to form LBP code (0-255 for 8 neighbors)
 * </pre>
 *
 * <h3>Advantages:</h3>
 * <ul>
 *   <li>No training required - features computed directly</li>
 *   <li>Robust to illumination changes</li>
 *   <li>Captures local texture patterns</li>
 *   <li>Computationally efficient</li>
 * </ul>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see FeatureExtractor
 */
public class LBPHExtractor implements FeatureExtractor, Serializable {

    private static final long serialVersionUID = 2L;

    public static final String ALGORITHM_NAME = "LBPH";
    public static final int VERSION = 2;

    /** Number of histogram bins (2^8 = 256 for 8 neighbors). */
    private static final int NUM_BINS = 256;

    private final ExtractorConfig config;
    private final int gridX;
    private final int gridY;
    private final int radius;
    private final int neighbors;

    private boolean initialized;

    /**
     * Creates an LBPH extractor with default settings.
     * Default: 8x8 grid, radius=1, 8 neighbors.
     */
    public LBPHExtractor() {
        this(8, 8, 1, 8);
    }

    /**
     * Creates an LBPH extractor with custom grid and LBP parameters.
     *
     * @param gridX number of horizontal grid divisions
     * @param gridY number of vertical grid divisions
     * @param radius radius for neighbor sampling
     * @param neighbors number of neighbors to sample
     */
    public LBPHExtractor(int gridX, int gridY, int radius, int neighbors) {
        if (gridX < 1 || gridY < 1) {
            throw new IllegalArgumentException("Grid dimensions must be positive");
        }
        if (radius < 1) {
            throw new IllegalArgumentException("Radius must be positive");
        }
        if (neighbors != 8) {
            throw new IllegalArgumentException("Currently only 8 neighbors supported");
        }

        this.config = new ExtractorConfig()
            .setNumComponents(gridX * gridY * NUM_BINS);
        this.gridX = gridX;
        this.gridY = gridY;
        this.radius = radius;
        this.neighbors = neighbors;
        this.initialized = true; // LBPH doesn't need training
    }

    @Override
    public void train(List<FaceImage> faces, List<String> labels) {
        // LBPH doesn't require training - features are computed directly
        // We just validate the input
        if (faces == null || faces.isEmpty()) {
            throw new IllegalArgumentException("Face list cannot be empty");
        }
        initialized = true;
    }

    @Override
    public boolean isTrained() {
        return initialized;
    }

    @Override
    public FeatureVector extract(FaceImage face) {
        if (!initialized) {
            throw new IllegalStateException("Extractor not initialized");
        }

        // Resize to expected dimensions
        FaceImage resized = face.getWidth() != config.getImageWidth() ||
                           face.getHeight() != config.getImageHeight()
            ? face.resize(config.getImageWidth(), config.getImageHeight())
            : face;

        // Get grayscale image
        int width = resized.getWidth();
        int height = resized.getHeight();
        int[][] grayImage = toGrayscaleMatrix(resized);

        // Compute LBP image
        int[][] lbpImage = computeLBP(grayImage, width, height);

        // Compute histograms for each region
        double[] histogram = computeRegionHistograms(lbpImage, width, height);

        // Normalize histogram
        if (config.isNormalize()) {
            normalizeHistogram(histogram);
        }

        return new FeatureVector(histogram, ALGORITHM_NAME, VERSION);
    }

    private int[][] toGrayscaleMatrix(FaceImage face) {
        int width = face.getWidth();
        int height = face.getHeight();
        int[][] gray = new int[height][width];

        double[] pixels = face.toGrayscaleArray();
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                gray[y][x] = (int) pixels[y * width + x];
            }
        }
        return gray;
    }

    private int[][] computeLBP(int[][] gray, int width, int height) {
        int[][] lbp = new int[height][width];

        // Precompute neighbor offsets for circular sampling
        double[] dx = new double[neighbors];
        double[] dy = new double[neighbors];
        for (int i = 0; i < neighbors; i++) {
            double angle = 2.0 * Math.PI * i / neighbors;
            dx[i] = radius * Math.cos(angle);
            dy[i] = -radius * Math.sin(angle);
        }

        for (int y = radius; y < height - radius; y++) {
            for (int x = radius; x < width - radius; x++) {
                int center = gray[y][x];
                int lbpCode = 0;

                for (int i = 0; i < neighbors; i++) {
                    // Bilinear interpolation for sub-pixel sampling
                    double nx = x + dx[i];
                    double ny = y + dy[i];

                    int neighborValue = bilinearInterpolate(gray, nx, ny, width, height);

                    if (neighborValue >= center) {
                        lbpCode |= (1 << i);
                    }
                }

                lbp[y][x] = lbpCode;
            }
        }

        return lbp;
    }

    private int bilinearInterpolate(int[][] gray, double x, double y, int width, int height) {
        int x1 = (int) Math.floor(x);
        int x2 = (int) Math.ceil(x);
        int y1 = (int) Math.floor(y);
        int y2 = (int) Math.ceil(y);

        // Clamp to image bounds
        x1 = Math.max(0, Math.min(x1, width - 1));
        x2 = Math.max(0, Math.min(x2, width - 1));
        y1 = Math.max(0, Math.min(y1, height - 1));
        y2 = Math.max(0, Math.min(y2, height - 1));

        if (x1 == x2 && y1 == y2) {
            return gray[y1][x1];
        }

        double fx = x - x1;
        double fy = y - y1;

        double val = (1 - fx) * (1 - fy) * gray[y1][x1]
                   + fx * (1 - fy) * gray[y1][x2]
                   + (1 - fx) * fy * gray[y2][x1]
                   + fx * fy * gray[y2][x2];

        return (int) Math.round(val);
    }

    private double[] computeRegionHistograms(int[][] lbp, int width, int height) {
        int regionWidth = width / gridX;
        int regionHeight = height / gridY;
        double[] histogram = new double[gridX * gridY * NUM_BINS];

        for (int gy = 0; gy < gridY; gy++) {
            for (int gx = 0; gx < gridX; gx++) {
                int regionIdx = (gy * gridX + gx) * NUM_BINS;

                int startX = gx * regionWidth;
                int startY = gy * regionHeight;
                int endX = (gx == gridX - 1) ? width : startX + regionWidth;
                int endY = (gy == gridY - 1) ? height : startY + regionHeight;

                // Build histogram for this region
                for (int y = Math.max(radius, startY); y < Math.min(height - radius, endY); y++) {
                    for (int x = Math.max(radius, startX); x < Math.min(width - radius, endX); x++) {
                        histogram[regionIdx + lbp[y][x]]++;
                    }
                }
            }
        }

        return histogram;
    }

    private void normalizeHistogram(double[] histogram) {
        // Normalize each region's histogram independently
        for (int region = 0; region < gridX * gridY; region++) {
            int start = region * NUM_BINS;
            double sum = 0;

            for (int i = 0; i < NUM_BINS; i++) {
                sum += histogram[start + i];
            }

            if (sum > 0) {
                for (int i = 0; i < NUM_BINS; i++) {
                    histogram[start + i] /= sum;
                }
            }
        }
    }

    @Override
    public int getFeatureDimension() {
        return gridX * gridY * NUM_BINS;
    }

    @Override
    public String getAlgorithmName() {
        return ALGORITHM_NAME;
    }

    @Override
    public int getVersion() {
        return VERSION;
    }

    @Override
    public int[] getExpectedImageSize() {
        return new int[]{config.getImageWidth(), config.getImageHeight()};
    }

    @Override
    public void reset() {
        // Nothing to reset for LBPH
        initialized = true;
    }

    @Override
    public ExtractorConfig getConfig() {
        return config;
    }

    /**
     * Gets the grid dimensions.
     *
     * @return array of [gridX, gridY]
     */
    public int[] getGridSize() {
        return new int[]{gridX, gridY};
    }

    /**
     * Gets the LBP radius.
     *
     * @return the radius
     */
    public int getRadius() {
        return radius;
    }

    /**
     * Gets the number of neighbors used in LBP.
     *
     * @return the neighbor count
     */
    public int getNeighbors() {
        return neighbors;
    }

    /**
     * Computes the uniform LBP pattern.
     * Uniform patterns have at most 2 bitwise transitions.
     *
     * @param lbpCode the LBP code
     * @return true if the pattern is uniform
     */
    public static boolean isUniform(int lbpCode) {
        int transitions = 0;
        int prev = lbpCode & 1;

        for (int i = 1; i < 8; i++) {
            int curr = (lbpCode >> i) & 1;
            if (curr != prev) {
                transitions++;
            }
            prev = curr;
        }

        // Check wrap-around transition
        if (((lbpCode >> 7) & 1) != (lbpCode & 1)) {
            transitions++;
        }

        return transitions <= 2;
    }

    @Override
    public String toString() {
        return String.format("LBPHExtractor{grid=%dx%d, radius=%d, neighbors=%d, dim=%d}",
            gridX, gridY, radius, neighbors, getFeatureDimension());
    }
}
