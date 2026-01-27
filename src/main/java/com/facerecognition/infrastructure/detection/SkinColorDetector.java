package com.facerecognition.infrastructure.detection;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FaceLandmarks;
import com.facerecognition.domain.model.FaceRegion;
import com.facerecognition.domain.service.FaceDetector;

import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.*;

/**
 * Skin color-based face detector using HSV color space analysis.
 *
 * <p>This detector uses skin color segmentation as a fast fallback method
 * for face detection. It employs:</p>
 * <ul>
 *   <li><b>HSV Color Space</b>: Better separation of chrominance and luminance</li>
 *   <li><b>Adaptive Thresholds</b>: Adjusts to image lighting conditions</li>
 *   <li><b>Connected Component Analysis</b>: Groups skin pixels into regions</li>
 *   <li><b>Morphological Operations</b>: Erosion/dilation for noise removal</li>
 * </ul>
 *
 * <h3>Algorithm Overview:</h3>
 * <ol>
 *   <li>Convert RGB image to HSV color space</li>
 *   <li>Apply skin color thresholds (with adaptive adjustment)</li>
 *   <li>Apply morphological operations (erosion then dilation)</li>
 *   <li>Find connected components in the binary mask</li>
 *   <li>Filter components by size and aspect ratio</li>
 *   <li>Return bounding boxes as face candidates</li>
 * </ol>
 *
 * <h3>Skin Color Model (HSV):</h3>
 * <pre>
 * Hue (H):        0-50 degrees (red/orange/yellow range)
 * Saturation (S): 0.1-0.7 (avoid very low/high saturation)
 * Value (V):      0.2-0.95 (avoid very dark/bright regions)
 *
 * Additional YCbCr checks can improve accuracy across skin tones.
 * </pre>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * SkinColorDetector detector = new SkinColorDetector();
 * detector.setMinFaceSize(50);
 * detector.setAdaptiveThreshold(true);
 *
 * List<FaceRegion> faces = detector.detectFaces(image);
 * }</pre>
 *
 * <p><b>Limitations:</b></p>
 * <ul>
 *   <li>Sensitive to lighting conditions and color temperature</li>
 *   <li>May produce false positives on skin-colored backgrounds</li>
 *   <li>Best used as a fallback or in conjunction with other detectors</li>
 * </ul>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see FaceDetector
 * @see CompositeFaceDetector
 */
public class SkinColorDetector implements FaceDetector, Serializable {

    private static final long serialVersionUID = 1L;

    /** Algorithm name for identification. */
    public static final String ALGORITHM_NAME = "SkinColor";

    /** Current algorithm version. */
    public static final String VERSION = "2.0";

    /** Default minimum face size in pixels. */
    public static final int DEFAULT_MIN_FACE_SIZE = 30;

    /** Default minimum confidence threshold. */
    public static final double DEFAULT_MIN_CONFIDENCE = 0.3;

    // HSV skin color range defaults (normalized 0-1)
    private static final double DEFAULT_HUE_MIN = 0.0;    // 0 degrees
    private static final double DEFAULT_HUE_MAX = 0.14;   // ~50 degrees (normalized to 0-1)
    private static final double DEFAULT_SAT_MIN = 0.1;
    private static final double DEFAULT_SAT_MAX = 0.7;
    private static final double DEFAULT_VAL_MIN = 0.2;
    private static final double DEFAULT_VAL_MAX = 0.95;

    // YCbCr range for additional skin tone verification
    private static final int YCBCR_CB_MIN = 77;
    private static final int YCBCR_CB_MAX = 127;
    private static final int YCBCR_CR_MIN = 133;
    private static final int YCBCR_CR_MAX = 173;

    // Configuration parameters
    private int minFaceSize;
    private int maxFaceSize;
    private double minConfidence;
    private boolean useAdaptiveThreshold;
    private boolean useYCbCrVerification;
    private int erosionSize;
    private int dilationSize;

    // Skin color thresholds (can be adjusted)
    private double hueMin;
    private double hueMax;
    private double satMin;
    private double satMax;
    private double valMin;
    private double valMax;

    // Aspect ratio constraints for face regions
    private static final double MIN_ASPECT_RATIO = 0.5;  // width/height
    private static final double MAX_ASPECT_RATIO = 2.0;

    // Minimum fill ratio (skin pixels / bounding box area)
    private static final double MIN_FILL_RATIO = 0.3;

    /**
     * Creates a skin color detector with default parameters.
     *
     * <p>Default settings:</p>
     * <ul>
     *   <li>Minimum face size: 30 pixels</li>
     *   <li>Adaptive threshold: enabled</li>
     *   <li>YCbCr verification: enabled</li>
     *   <li>Erosion/Dilation: 3x3 kernel</li>
     * </ul>
     */
    public SkinColorDetector() {
        this.minFaceSize = DEFAULT_MIN_FACE_SIZE;
        this.maxFaceSize = Integer.MAX_VALUE;
        this.minConfidence = DEFAULT_MIN_CONFIDENCE;
        this.useAdaptiveThreshold = true;
        this.useYCbCrVerification = true;
        this.erosionSize = 3;
        this.dilationSize = 5;

        // Initialize default HSV thresholds
        resetThresholdsToDefault();
    }

    /**
     * Resets skin color thresholds to default values.
     */
    public void resetThresholdsToDefault() {
        this.hueMin = DEFAULT_HUE_MIN;
        this.hueMax = DEFAULT_HUE_MAX;
        this.satMin = DEFAULT_SAT_MIN;
        this.satMax = DEFAULT_SAT_MAX;
        this.valMin = DEFAULT_VAL_MIN;
        this.valMax = DEFAULT_VAL_MAX;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Detection process:</p>
     * <ol>
     *   <li>Convert to HSV and create skin mask</li>
     *   <li>Apply morphological operations</li>
     *   <li>Find and filter connected components</li>
     *   <li>Return face region candidates</li>
     * </ol>
     */
    @Override
    public List<FaceRegion> detectFaces(FaceImage image) {
        return detectFaces(image, minConfidence);
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public List<FaceRegion> detectFaces(FaceImage image, double minConfidence) {
        Objects.requireNonNull(image, "Image cannot be null");
        if (minConfidence < 0.0 || minConfidence > 1.0) {
            throw new IllegalArgumentException("Confidence threshold must be between 0.0 and 1.0");
        }

        BufferedImage bufferedImage = image.getImage();
        int width = bufferedImage.getWidth();
        int height = bufferedImage.getHeight();

        // Create skin color mask
        boolean[][] skinMask = createSkinMask(bufferedImage);

        // Apply morphological operations
        skinMask = applyErosion(skinMask, erosionSize);
        skinMask = applyDilation(skinMask, dilationSize);

        // Find connected components
        List<ConnectedComponent> components = findConnectedComponents(skinMask);

        // Filter and convert to face regions
        List<FaceRegion> faces = new ArrayList<>();
        int effectiveMaxSize = Math.min(maxFaceSize, Math.min(width, height));

        for (ConnectedComponent component : components) {
            // Check size constraints
            int regionWidth = component.maxX - component.minX + 1;
            int regionHeight = component.maxY - component.minY + 1;
            int regionSize = Math.max(regionWidth, regionHeight);

            if (regionSize < minFaceSize || regionSize > effectiveMaxSize) {
                continue;
            }

            // Check aspect ratio
            double aspectRatio = (double) regionWidth / regionHeight;
            if (aspectRatio < MIN_ASPECT_RATIO || aspectRatio > MAX_ASPECT_RATIO) {
                continue;
            }

            // Check fill ratio
            double fillRatio = (double) component.pixelCount / (regionWidth * regionHeight);
            if (fillRatio < MIN_FILL_RATIO) {
                continue;
            }

            // Calculate confidence based on multiple factors
            double confidence = calculateConfidence(component, regionWidth, regionHeight, fillRatio);

            if (confidence >= minConfidence) {
                faces.add(new FaceRegion(
                    component.minX, component.minY,
                    regionWidth, regionHeight,
                    confidence
                ));
            }
        }

        // Merge overlapping detections
        faces = mergeOverlappingRegions(faces);

        // Sort by confidence
        faces.sort((a, b) -> Double.compare(b.getConfidence(), a.getConfidence()));

        return faces;
    }

    /**
     * Creates a binary skin color mask from the input image.
     *
     * <p>Uses HSV thresholds with optional adaptive adjustment and
     * YCbCr verification for improved accuracy across skin tones.</p>
     *
     * @param image the input image
     * @return binary mask (true = skin pixel)
     */
    private boolean[][] createSkinMask(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        boolean[][] mask = new boolean[height][width];

        // Compute adaptive thresholds if enabled
        double[] adaptiveParams = useAdaptiveThreshold ?
            computeAdaptiveThresholds(image) : null;

        double adaptedHueMin = hueMin;
        double adaptedHueMax = hueMax;
        double adaptedSatMin = satMin;
        double adaptedSatMax = satMax;
        double adaptedValMin = valMin;
        double adaptedValMax = valMax;

        if (adaptiveParams != null) {
            // Adjust thresholds based on image statistics
            adaptedHueMin = Math.max(0, hueMin - adaptiveParams[0]);
            adaptedHueMax = Math.min(1, hueMax + adaptiveParams[0]);
            adaptedSatMin = Math.max(0, satMin - adaptiveParams[1]);
            adaptedSatMax = Math.min(1, satMax + adaptiveParams[1]);
            adaptedValMin = Math.max(0, valMin - adaptiveParams[2]);
            adaptedValMax = Math.min(1, valMax + adaptiveParams[2]);
        }

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;

                // Convert to HSV
                double[] hsv = rgbToHsv(r, g, b);
                double h = hsv[0];
                double s = hsv[1];
                double v = hsv[2];

                // Check HSV thresholds
                boolean isSkinHsv = checkHsvThresholds(h, s, v,
                    adaptedHueMin, adaptedHueMax,
                    adaptedSatMin, adaptedSatMax,
                    adaptedValMin, adaptedValMax);

                // Optional YCbCr verification
                boolean isSkinYcbcr = true;
                if (useYCbCrVerification && isSkinHsv) {
                    int[] ycbcr = rgbToYcbcr(r, g, b);
                    isSkinYcbcr = checkYcbcrThresholds(ycbcr[1], ycbcr[2]);
                }

                mask[y][x] = isSkinHsv && isSkinYcbcr;
            }
        }

        return mask;
    }

    /**
     * Converts RGB to HSV color space.
     *
     * @param r red component (0-255)
     * @param g green component (0-255)
     * @param b blue component (0-255)
     * @return array of [H, S, V] with H in 0-1, S in 0-1, V in 0-1
     */
    private double[] rgbToHsv(int r, int g, int b) {
        double rNorm = r / 255.0;
        double gNorm = g / 255.0;
        double bNorm = b / 255.0;

        double max = Math.max(Math.max(rNorm, gNorm), bNorm);
        double min = Math.min(Math.min(rNorm, gNorm), bNorm);
        double delta = max - min;

        // Value
        double v = max;

        // Saturation
        double s = (max == 0) ? 0 : delta / max;

        // Hue
        double h = 0;
        if (delta != 0) {
            if (max == rNorm) {
                h = (gNorm - bNorm) / delta;
                if (g < b) {
                    h += 6;
                }
            } else if (max == gNorm) {
                h = 2 + (bNorm - rNorm) / delta;
            } else {
                h = 4 + (rNorm - gNorm) / delta;
            }
            h /= 6; // Normalize to 0-1
        }

        return new double[]{h, s, v};
    }

    /**
     * Converts RGB to YCbCr color space.
     *
     * @param r red component (0-255)
     * @param g green component (0-255)
     * @param b blue component (0-255)
     * @return array of [Y, Cb, Cr]
     */
    private int[] rgbToYcbcr(int r, int g, int b) {
        int y = (int) (0.299 * r + 0.587 * g + 0.114 * b);
        int cb = (int) (128 - 0.169 * r - 0.331 * g + 0.500 * b);
        int cr = (int) (128 + 0.500 * r - 0.419 * g - 0.081 * b);

        return new int[]{y, cb, cr};
    }

    /**
     * Checks if HSV values fall within skin color thresholds.
     */
    private boolean checkHsvThresholds(double h, double s, double v,
                                       double hMin, double hMax,
                                       double sMin, double sMax,
                                       double vMin, double vMax) {
        // Handle hue wrap-around for red tones
        boolean hueInRange;
        if (hMin <= hMax) {
            hueInRange = h >= hMin && h <= hMax;
        } else {
            // Wrap-around case (red hue spans 0)
            hueInRange = h >= hMin || h <= hMax;
        }

        // Also check for red tones at high end of hue (>0.9)
        boolean isRedTone = h > 0.9 && h <= 1.0;

        return (hueInRange || isRedTone) &&
               s >= sMin && s <= sMax &&
               v >= vMin && v <= vMax;
    }

    /**
     * Checks if YCbCr values fall within skin color thresholds.
     */
    private boolean checkYcbcrThresholds(int cb, int cr) {
        return cb >= YCBCR_CB_MIN && cb <= YCBCR_CB_MAX &&
               cr >= YCBCR_CR_MIN && cr <= YCBCR_CR_MAX;
    }

    /**
     * Computes adaptive threshold adjustments based on image statistics.
     *
     * <p>Analyzes image brightness and color distribution to adjust
     * thresholds for varying lighting conditions.</p>
     *
     * @param image the input image
     * @return array of [hue_adjust, sat_adjust, val_adjust]
     */
    private double[] computeAdaptiveThresholds(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int sampleSize = Math.min(10000, width * height);
        int step = Math.max(1, (width * height) / sampleSize);

        double sumBrightness = 0;
        double sumSaturation = 0;
        int count = 0;

        for (int i = 0; i < width * height; i += step) {
            int x = i % width;
            int y = i / width;

            int rgb = image.getRGB(x, y);
            int r = (rgb >> 16) & 0xFF;
            int g = (rgb >> 8) & 0xFF;
            int b = rgb & 0xFF;

            double[] hsv = rgbToHsv(r, g, b);
            sumBrightness += hsv[2];
            sumSaturation += hsv[1];
            count++;
        }

        double avgBrightness = sumBrightness / count;
        double avgSaturation = sumSaturation / count;

        // Adjust thresholds based on average image properties
        double hueAdjust = 0.02; // Small fixed adjustment

        // Darker images need wider value range
        double valAdjust = (avgBrightness < 0.4) ? 0.1 : 0.0;

        // Lower saturation images need wider saturation range
        double satAdjust = (avgSaturation < 0.3) ? 0.1 : 0.0;

        return new double[]{hueAdjust, satAdjust, valAdjust};
    }

    /**
     * Applies morphological erosion to the binary mask.
     *
     * <p>Erosion removes small noise and separates weakly connected regions.</p>
     *
     * @param mask the input binary mask
     * @param kernelSize the erosion kernel size (must be odd)
     * @return the eroded mask
     */
    private boolean[][] applyErosion(boolean[][] mask, int kernelSize) {
        if (kernelSize <= 0) {
            return mask;
        }

        int height = mask.length;
        int width = mask[0].length;
        boolean[][] result = new boolean[height][width];
        int radius = kernelSize / 2;

        for (int y = radius; y < height - radius; y++) {
            for (int x = radius; x < width - radius; x++) {
                // Check if all pixels in kernel are true
                boolean allTrue = true;
                outer:
                for (int ky = -radius; ky <= radius && allTrue; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        if (!mask[y + ky][x + kx]) {
                            allTrue = false;
                            break outer;
                        }
                    }
                }
                result[y][x] = allTrue;
            }
        }

        return result;
    }

    /**
     * Applies morphological dilation to the binary mask.
     *
     * <p>Dilation fills small holes and connects nearby regions.</p>
     *
     * @param mask the input binary mask
     * @param kernelSize the dilation kernel size (must be odd)
     * @return the dilated mask
     */
    private boolean[][] applyDilation(boolean[][] mask, int kernelSize) {
        if (kernelSize <= 0) {
            return mask;
        }

        int height = mask.length;
        int width = mask[0].length;
        boolean[][] result = new boolean[height][width];
        int radius = kernelSize / 2;

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                // Check if any pixel in kernel is true
                boolean anyTrue = false;
                outer:
                for (int ky = -radius; ky <= radius && !anyTrue; ky++) {
                    for (int kx = -radius; kx <= radius; kx++) {
                        int ny = y + ky;
                        int nx = x + kx;
                        if (ny >= 0 && ny < height && nx >= 0 && nx < width) {
                            if (mask[ny][nx]) {
                                anyTrue = true;
                                break outer;
                            }
                        }
                    }
                }
                result[y][x] = anyTrue;
            }
        }

        return result;
    }

    /**
     * Finds connected components in the binary mask using flood fill.
     *
     * @param mask the binary skin mask
     * @return list of connected components
     */
    private List<ConnectedComponent> findConnectedComponents(boolean[][] mask) {
        int height = mask.length;
        int width = mask[0].length;
        boolean[][] visited = new boolean[height][width];
        List<ConnectedComponent> components = new ArrayList<>();

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                if (mask[y][x] && !visited[y][x]) {
                    ConnectedComponent component = floodFill(mask, visited, x, y);
                    if (component.pixelCount >= 100) { // Minimum pixel count
                        components.add(component);
                    }
                }
            }
        }

        return components;
    }

    /**
     * Performs flood fill to extract a connected component.
     *
     * @param mask the binary mask
     * @param visited tracking array for visited pixels
     * @param startX starting x coordinate
     * @param startY starting y coordinate
     * @return the connected component
     */
    private ConnectedComponent floodFill(boolean[][] mask, boolean[][] visited,
                                         int startX, int startY) {
        int height = mask.length;
        int width = mask[0].length;

        ConnectedComponent component = new ConnectedComponent();
        Queue<int[]> queue = new LinkedList<>();
        queue.add(new int[]{startX, startY});
        visited[startY][startX] = true;

        // 4-connectivity neighbors
        int[] dx = {0, 1, 0, -1};
        int[] dy = {-1, 0, 1, 0};

        while (!queue.isEmpty()) {
            int[] pixel = queue.poll();
            int x = pixel[0];
            int y = pixel[1];

            // Update component bounds
            component.minX = Math.min(component.minX, x);
            component.maxX = Math.max(component.maxX, x);
            component.minY = Math.min(component.minY, y);
            component.maxY = Math.max(component.maxY, y);
            component.pixelCount++;
            component.sumX += x;
            component.sumY += y;

            // Check neighbors
            for (int i = 0; i < 4; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];

                if (nx >= 0 && nx < width && ny >= 0 && ny < height &&
                    mask[ny][nx] && !visited[ny][nx]) {
                    visited[ny][nx] = true;
                    queue.add(new int[]{nx, ny});
                }
            }
        }

        // Calculate centroid
        component.centroidX = (int) (component.sumX / component.pixelCount);
        component.centroidY = (int) (component.sumY / component.pixelCount);

        return component;
    }

    /**
     * Calculates confidence score for a detected region.
     *
     * <p>Confidence is based on:</p>
     * <ul>
     *   <li>Fill ratio (higher is better)</li>
     *   <li>Aspect ratio (closer to 1.0 is better)</li>
     *   <li>Size (medium sizes score higher)</li>
     * </ul>
     *
     * @param component the connected component
     * @param width region width
     * @param height region height
     * @param fillRatio skin pixel fill ratio
     * @return confidence score (0.0 to 1.0)
     */
    private double calculateConfidence(ConnectedComponent component,
                                       int width, int height, double fillRatio) {
        // Fill ratio contribution (0.4 weight)
        double fillScore = Math.min(fillRatio / 0.6, 1.0) * 0.4;

        // Aspect ratio contribution (0.3 weight) - optimal around 0.75-1.0
        double aspectRatio = (double) width / height;
        double optimalAspect = 0.85;
        double aspectDeviation = Math.abs(aspectRatio - optimalAspect);
        double aspectScore = Math.max(0, 1.0 - aspectDeviation) * 0.3;

        // Size contribution (0.3 weight) - prefer medium-sized regions
        int regionSize = Math.max(width, height);
        double sizeScore;
        if (regionSize < minFaceSize * 2) {
            sizeScore = 0.7 * 0.3; // Small regions get lower score
        } else if (regionSize < minFaceSize * 5) {
            sizeScore = 1.0 * 0.3; // Medium regions get full score
        } else {
            sizeScore = 0.8 * 0.3; // Large regions get slightly lower score
        }

        return Math.min(1.0, fillScore + aspectScore + sizeScore);
    }

    /**
     * Merges overlapping face region detections.
     *
     * @param regions list of candidate regions
     * @return merged list of regions
     */
    private List<FaceRegion> mergeOverlappingRegions(List<FaceRegion> regions) {
        if (regions.size() <= 1) {
            return regions;
        }

        List<FaceRegion> merged = new ArrayList<>();
        boolean[] used = new boolean[regions.size()];

        for (int i = 0; i < regions.size(); i++) {
            if (used[i]) {
                continue;
            }

            FaceRegion current = regions.get(i);
            double maxConfidence = current.getConfidence();

            // Find overlapping regions
            for (int j = i + 1; j < regions.size(); j++) {
                if (used[j]) {
                    continue;
                }

                FaceRegion other = regions.get(j);
                double iou = current.intersectionOverUnion(other);

                if (iou > 0.3) {
                    used[j] = true;
                    maxConfidence = Math.max(maxConfidence, other.getConfidence());
                }
            }

            // Create merged region with boosted confidence
            merged.add(new FaceRegion(
                current.getX(), current.getY(),
                current.getWidth(), current.getHeight(),
                Math.min(1.0, maxConfidence * 1.1)
            ));
        }

        return merged;
    }

    /**
     * {@inheritDoc}
     *
     * <p>This implementation does not support landmark detection.</p>
     */
    @Override
    public boolean supportsLandmarks() {
        return false;
    }

    /**
     * {@inheritDoc}
     *
     * @return always empty as landmarks are not supported
     */
    @Override
    public Optional<FaceLandmarks> detectLandmarks(FaceImage image, FaceRegion faceRegion) {
        return Optional.empty();
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getName() {
        return ALGORITHM_NAME;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public String getVersion() {
        return VERSION;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public int getMinFaceSize() {
        return minFaceSize;
    }

    /**
     * {@inheritDoc}
     */
    @Override
    public void setMinFaceSize(int minSize) {
        if (minSize < 10) {
            throw new IllegalArgumentException("Minimum face size must be at least 10 pixels");
        }
        this.minFaceSize = minSize;
    }

    /**
     * Gets the maximum face size to detect.
     *
     * @return the maximum face size in pixels
     */
    public int getMaxFaceSize() {
        return maxFaceSize;
    }

    /**
     * Sets the maximum face size to detect.
     *
     * @param maxSize the maximum size in pixels
     */
    public void setMaxFaceSize(int maxSize) {
        if (maxSize < minFaceSize) {
            throw new IllegalArgumentException("Maximum size must be >= minimum size");
        }
        this.maxFaceSize = maxSize;
    }

    /**
     * Checks if adaptive thresholding is enabled.
     *
     * @return true if adaptive thresholding is enabled
     */
    public boolean isAdaptiveThresholdEnabled() {
        return useAdaptiveThreshold;
    }

    /**
     * Enables or disables adaptive thresholding.
     *
     * @param enabled true to enable adaptive thresholding
     */
    public void setAdaptiveThreshold(boolean enabled) {
        this.useAdaptiveThreshold = enabled;
    }

    /**
     * Checks if YCbCr verification is enabled.
     *
     * @return true if YCbCr verification is enabled
     */
    public boolean isYCbCrVerificationEnabled() {
        return useYCbCrVerification;
    }

    /**
     * Enables or disables YCbCr color space verification.
     *
     * @param enabled true to enable YCbCr verification
     */
    public void setYCbCrVerification(boolean enabled) {
        this.useYCbCrVerification = enabled;
    }

    /**
     * Sets the morphological operation kernel sizes.
     *
     * @param erosion erosion kernel size (odd number, 0 to disable)
     * @param dilation dilation kernel size (odd number, 0 to disable)
     */
    public void setMorphologyKernels(int erosion, int dilation) {
        if (erosion < 0 || dilation < 0) {
            throw new IllegalArgumentException("Kernel sizes cannot be negative");
        }
        this.erosionSize = erosion;
        this.dilationSize = dilation;
    }

    /**
     * Sets custom HSV skin color thresholds.
     *
     * @param hueMin minimum hue (0-1)
     * @param hueMax maximum hue (0-1)
     * @param satMin minimum saturation (0-1)
     * @param satMax maximum saturation (0-1)
     * @param valMin minimum value (0-1)
     * @param valMax maximum value (0-1)
     */
    public void setHsvThresholds(double hueMin, double hueMax,
                                  double satMin, double satMax,
                                  double valMin, double valMax) {
        validateRange("Hue min", hueMin, 0.0, 1.0);
        validateRange("Hue max", hueMax, 0.0, 1.0);
        validateRange("Saturation min", satMin, 0.0, 1.0);
        validateRange("Saturation max", satMax, 0.0, 1.0);
        validateRange("Value min", valMin, 0.0, 1.0);
        validateRange("Value max", valMax, 0.0, 1.0);

        this.hueMin = hueMin;
        this.hueMax = hueMax;
        this.satMin = satMin;
        this.satMax = satMax;
        this.valMin = valMin;
        this.valMax = valMax;
    }

    /**
     * Validates a value is within a range.
     */
    private void validateRange(String name, double value, double min, double max) {
        if (value < min || value > max) {
            throw new IllegalArgumentException(
                String.format("%s must be between %.1f and %.1f", name, min, max));
        }
    }

    @Override
    public String toString() {
        return String.format(
            "SkinColorDetector{minSize=%d, adaptive=%s, ycbcr=%s, erosion=%d, dilation=%d}",
            minFaceSize, useAdaptiveThreshold, useYCbCrVerification, erosionSize, dilationSize
        );
    }

    // =========================================================================
    // Inner Classes
    // =========================================================================

    /**
     * Represents a connected component of skin pixels.
     */
    private static class ConnectedComponent {
        int minX = Integer.MAX_VALUE;
        int maxX = Integer.MIN_VALUE;
        int minY = Integer.MAX_VALUE;
        int maxY = Integer.MIN_VALUE;
        int pixelCount = 0;
        long sumX = 0;
        long sumY = 0;
        int centroidX;
        int centroidY;
    }
}
