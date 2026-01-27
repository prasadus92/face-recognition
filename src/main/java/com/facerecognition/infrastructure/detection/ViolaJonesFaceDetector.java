package com.facerecognition.infrastructure.detection;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FaceLandmarks;
import com.facerecognition.domain.model.FaceRegion;
import com.facerecognition.domain.service.FaceDetector;

import java.awt.image.BufferedImage;
import java.io.Serializable;
import java.util.*;

/**
 * Pure Java implementation of the Viola-Jones face detection algorithm.
 *
 * <p>The Viola-Jones algorithm is a machine learning based approach for
 * rapid object detection. This implementation uses:</p>
 * <ul>
 *   <li><b>Integral Images</b>: For O(1) computation of rectangular features</li>
 *   <li><b>Haar-like Features</b>: Edge, line, and center-surround features</li>
 *   <li><b>Sliding Window</b>: Multi-scale detection across the image</li>
 *   <li><b>Simple Cascade</b>: Basic classifier chain for efficiency</li>
 * </ul>
 *
 * <h3>Algorithm Overview:</h3>
 * <ol>
 *   <li>Compute integral image for fast feature calculation</li>
 *   <li>Apply sliding window at multiple scales</li>
 *   <li>Evaluate Haar-like features at each window position</li>
 *   <li>Apply cascade of weak classifiers</li>
 *   <li>Merge overlapping detections using non-maximum suppression</li>
 * </ol>
 *
 * <h3>Integral Image Computation:</h3>
 * <pre>
 * ii(x,y) = sum of pixels in rectangle from (0,0) to (x,y)
 * ii(x,y) = i(x,y) + ii(x-1,y) + ii(x,y-1) - ii(x-1,y-1)
 *
 * Rectangle sum = ii(D) - ii(B) - ii(C) + ii(A)
 * where A,B,C,D are corners of the rectangle
 * </pre>
 *
 * <h3>Haar-like Features:</h3>
 * <pre>
 * Two-rectangle (edge):    Three-rectangle (line):
 * +-------+-------+        +---+---+---+
 * | white | black |        | w | b | w |
 * +-------+-------+        +---+---+---+
 * </pre>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * ViolaJonesFaceDetector detector = new ViolaJonesFaceDetector();
 * detector.setMinFaceSize(60);
 * detector.setScaleFactor(1.2);
 *
 * List<FaceRegion> faces = detector.detectFaces(image);
 * for (FaceRegion face : faces) {
 *     System.out.println("Found face at: " + face);
 * }
 * }</pre>
 *
 * <p><b>Note:</b> This is a simplified implementation for educational purposes.
 * For production use, consider using trained cascade classifiers from OpenCV
 * or dedicated machine learning libraries.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see FaceDetector
 * @see FaceRegion
 */
public class ViolaJonesFaceDetector implements FaceDetector, Serializable {

    private static final long serialVersionUID = 1L;

    /** Algorithm name for identification. */
    public static final String ALGORITHM_NAME = "ViolaJones";

    /** Current algorithm version. */
    public static final String VERSION = "2.0";

    /** Default minimum face size in pixels. */
    public static final int DEFAULT_MIN_FACE_SIZE = 24;

    /** Default scale factor for multi-scale detection. */
    public static final double DEFAULT_SCALE_FACTOR = 1.1;

    /** Default minimum neighbors for detection merging. */
    public static final int DEFAULT_MIN_NEIGHBORS = 3;

    /** Default detection threshold. */
    public static final double DEFAULT_THRESHOLD = 0.5;

    /** IoU threshold for non-maximum suppression. */
    private static final double NMS_IOU_THRESHOLD = 0.3;

    /** Maximum aspect ratio deviation from 1.0 for face candidates. */
    private static final double MAX_ASPECT_RATIO_DEVIATION = 0.5;

    // Configuration parameters
    private int minFaceSize;
    private int maxFaceSize;
    private double scaleFactor;
    private int minNeighbors;
    private double detectionThreshold;

    // Pre-computed Haar feature templates
    private final List<HaarFeature> featureTemplates;

    // Weak classifier weights (simplified - in practice these would be learned)
    private final double[] classifierWeights;
    private final double[] classifierThresholds;

    /**
     * Creates a Viola-Jones face detector with default parameters.
     *
     * <p>Default settings:</p>
     * <ul>
     *   <li>Minimum face size: 24 pixels</li>
     *   <li>Scale factor: 1.1 (10% increase per scale)</li>
     *   <li>Minimum neighbors: 3</li>
     *   <li>Detection threshold: 0.5</li>
     * </ul>
     */
    public ViolaJonesFaceDetector() {
        this(DEFAULT_MIN_FACE_SIZE, DEFAULT_SCALE_FACTOR, DEFAULT_MIN_NEIGHBORS);
    }

    /**
     * Creates a Viola-Jones face detector with custom parameters.
     *
     * @param minFaceSize minimum face size to detect in pixels
     * @param scaleFactor scale factor for image pyramid (typical: 1.1-1.4)
     * @param minNeighbors minimum overlapping detections to accept a face
     * @throws IllegalArgumentException if parameters are invalid
     */
    public ViolaJonesFaceDetector(int minFaceSize, double scaleFactor, int minNeighbors) {
        validateParameters(minFaceSize, scaleFactor, minNeighbors);

        this.minFaceSize = minFaceSize;
        this.maxFaceSize = Integer.MAX_VALUE;
        this.scaleFactor = scaleFactor;
        this.minNeighbors = minNeighbors;
        this.detectionThreshold = DEFAULT_THRESHOLD;

        // Initialize Haar feature templates
        this.featureTemplates = initializeHaarFeatures();

        // Initialize classifier weights (simplified - would be learned in practice)
        this.classifierWeights = initializeClassifierWeights();
        this.classifierThresholds = initializeClassifierThresholds();
    }

    /**
     * Validates constructor parameters.
     *
     * @param minSize minimum face size
     * @param scale scale factor
     * @param neighbors minimum neighbors
     * @throws IllegalArgumentException if any parameter is invalid
     */
    private void validateParameters(int minSize, double scale, int neighbors) {
        if (minSize < 10) {
            throw new IllegalArgumentException("Minimum face size must be at least 10 pixels");
        }
        if (scale <= 1.0 || scale > 2.0) {
            throw new IllegalArgumentException("Scale factor must be between 1.0 (exclusive) and 2.0");
        }
        if (neighbors < 0) {
            throw new IllegalArgumentException("Minimum neighbors cannot be negative");
        }
    }

    /**
     * Initializes the set of Haar-like feature templates.
     *
     * <p>Creates a diverse set of features including:</p>
     * <ul>
     *   <li>Horizontal edge features (eye regions)</li>
     *   <li>Vertical edge features (nose bridge)</li>
     *   <li>Line features (eyebrows, mouth)</li>
     *   <li>Center-surround features (nose, eyes)</li>
     * </ul>
     *
     * @return list of Haar feature templates
     */
    private List<HaarFeature> initializeHaarFeatures() {
        List<HaarFeature> features = new ArrayList<>();

        // Two-rectangle horizontal edge features (good for eyes, eyebrows)
        features.add(new HaarFeature(HaarFeatureType.TWO_RECT_HORIZONTAL, 0, 4, 24, 8));
        features.add(new HaarFeature(HaarFeatureType.TWO_RECT_HORIZONTAL, 0, 6, 24, 6));
        features.add(new HaarFeature(HaarFeatureType.TWO_RECT_HORIZONTAL, 2, 8, 20, 4));
        features.add(new HaarFeature(HaarFeatureType.TWO_RECT_HORIZONTAL, 4, 4, 16, 8));

        // Two-rectangle vertical edge features (good for nose, face edges)
        features.add(new HaarFeature(HaarFeatureType.TWO_RECT_VERTICAL, 10, 0, 4, 24));
        features.add(new HaarFeature(HaarFeatureType.TWO_RECT_VERTICAL, 8, 4, 8, 16));
        features.add(new HaarFeature(HaarFeatureType.TWO_RECT_VERTICAL, 4, 2, 6, 20));
        features.add(new HaarFeature(HaarFeatureType.TWO_RECT_VERTICAL, 14, 2, 6, 20));

        // Three-rectangle horizontal line features (good for nose, mouth)
        features.add(new HaarFeature(HaarFeatureType.THREE_RECT_HORIZONTAL, 4, 12, 16, 6));
        features.add(new HaarFeature(HaarFeatureType.THREE_RECT_HORIZONTAL, 6, 14, 12, 4));
        features.add(new HaarFeature(HaarFeatureType.THREE_RECT_HORIZONTAL, 2, 16, 20, 4));

        // Three-rectangle vertical line features
        features.add(new HaarFeature(HaarFeatureType.THREE_RECT_VERTICAL, 10, 4, 4, 16));
        features.add(new HaarFeature(HaarFeatureType.THREE_RECT_VERTICAL, 8, 6, 8, 12));

        // Four-rectangle (checker) features (good for diagonal structures)
        features.add(new HaarFeature(HaarFeatureType.FOUR_RECT_CHECKER, 4, 4, 16, 16));
        features.add(new HaarFeature(HaarFeatureType.FOUR_RECT_CHECKER, 6, 6, 12, 12));
        features.add(new HaarFeature(HaarFeatureType.FOUR_RECT_CHECKER, 2, 2, 20, 20));

        // Eye region specific features
        features.add(new HaarFeature(HaarFeatureType.TWO_RECT_HORIZONTAL, 2, 4, 8, 6));
        features.add(new HaarFeature(HaarFeatureType.TWO_RECT_HORIZONTAL, 14, 4, 8, 6));

        // Mouth region features
        features.add(new HaarFeature(HaarFeatureType.THREE_RECT_HORIZONTAL, 4, 18, 16, 4));

        return features;
    }

    /**
     * Initializes simplified classifier weights.
     * In a real implementation, these would be learned via AdaBoost.
     *
     * @return array of classifier weights
     */
    private double[] initializeClassifierWeights() {
        double[] weights = new double[featureTemplates.size()];
        // Assign weights based on feature importance (simplified)
        for (int i = 0; i < weights.length; i++) {
            HaarFeature feature = featureTemplates.get(i);
            // Eye region features get higher weights
            if (feature.y >= 4 && feature.y <= 10) {
                weights[i] = 1.5;
            }
            // Nose/mouth features
            else if (feature.y > 10) {
                weights[i] = 1.0;
            } else {
                weights[i] = 0.8;
            }
        }
        return weights;
    }

    /**
     * Initializes classifier thresholds.
     * In a real implementation, these would be learned via AdaBoost.
     *
     * @return array of classifier thresholds
     */
    private double[] initializeClassifierThresholds() {
        double[] thresholds = new double[featureTemplates.size()];
        Arrays.fill(thresholds, 0.0); // Default threshold at 0
        return thresholds;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Detection process:</p>
     * <ol>
     *   <li>Convert image to grayscale</li>
     *   <li>Compute integral image</li>
     *   <li>Apply sliding window at multiple scales</li>
     *   <li>Evaluate Haar cascade at each position</li>
     *   <li>Merge overlapping detections</li>
     * </ol>
     */
    @Override
    public List<FaceRegion> detectFaces(FaceImage image) {
        return detectFaces(image, detectionThreshold);
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
        int imageWidth = bufferedImage.getWidth();
        int imageHeight = bufferedImage.getHeight();

        // Convert to grayscale
        int[][] grayscale = convertToGrayscale(bufferedImage);

        // Compute integral image
        long[][] integralImage = computeIntegralImage(grayscale);

        // Compute squared integral image for variance normalization
        long[][] squaredIntegralImage = computeSquaredIntegralImage(grayscale);

        // Collect all candidate detections
        List<Detection> candidates = new ArrayList<>();

        // Calculate effective max face size
        int effectiveMaxSize = Math.min(maxFaceSize, Math.min(imageWidth, imageHeight));

        // Multi-scale detection
        double currentScale = 1.0;
        int windowSize = minFaceSize;

        while (windowSize <= effectiveMaxSize) {
            // Calculate step size (typically 1-2 pixels at base scale, more at larger scales)
            int stepSize = Math.max(1, (int) (2 * currentScale));

            // Slide window across image
            for (int y = 0; y <= imageHeight - windowSize; y += stepSize) {
                for (int x = 0; x <= imageWidth - windowSize; x += stepSize) {
                    // Evaluate cascade classifier at this window
                    double confidence = evaluateCascade(
                        integralImage, squaredIntegralImage,
                        x, y, windowSize, currentScale
                    );

                    if (confidence >= minConfidence) {
                        candidates.add(new Detection(x, y, windowSize, windowSize, confidence));
                    }
                }
            }

            // Move to next scale
            currentScale *= scaleFactor;
            windowSize = (int) (minFaceSize * currentScale);
        }

        // Merge overlapping detections using non-maximum suppression
        List<FaceRegion> mergedDetections = mergeDetections(candidates);

        // Sort by confidence (highest first)
        mergedDetections.sort((a, b) -> Double.compare(b.getConfidence(), a.getConfidence()));

        return mergedDetections;
    }

    /**
     * Converts a BufferedImage to a grayscale pixel array.
     *
     * @param image the input image
     * @return 2D array of grayscale values (0-255)
     */
    private int[][] convertToGrayscale(BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        int[][] grayscale = new int[height][width];

        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int rgb = image.getRGB(x, y);
                int r = (rgb >> 16) & 0xFF;
                int g = (rgb >> 8) & 0xFF;
                int b = rgb & 0xFF;
                // Standard luminosity formula
                grayscale[y][x] = (int) (0.299 * r + 0.587 * g + 0.114 * b);
            }
        }

        return grayscale;
    }

    /**
     * Computes the integral image (summed area table).
     *
     * <p>The integral image allows O(1) computation of any rectangular sum.
     * Each pixel ii(x,y) contains the sum of all pixels above and to the left.</p>
     *
     * <p>Formula: ii(x,y) = i(x,y) + ii(x-1,y) + ii(x,y-1) - ii(x-1,y-1)</p>
     *
     * @param grayscale the grayscale image
     * @return the integral image
     */
    private long[][] computeIntegralImage(int[][] grayscale) {
        int height = grayscale.length;
        int width = grayscale[0].length;

        // Add 1 to dimensions for easier boundary handling
        long[][] integral = new long[height + 1][width + 1];

        for (int y = 1; y <= height; y++) {
            long rowSum = 0;
            for (int x = 1; x <= width; x++) {
                rowSum += grayscale[y - 1][x - 1];
                integral[y][x] = integral[y - 1][x] + rowSum;
            }
        }

        return integral;
    }

    /**
     * Computes the squared integral image for variance calculation.
     *
     * @param grayscale the grayscale image
     * @return the squared integral image
     */
    private long[][] computeSquaredIntegralImage(int[][] grayscale) {
        int height = grayscale.length;
        int width = grayscale[0].length;

        long[][] squaredIntegral = new long[height + 1][width + 1];

        for (int y = 1; y <= height; y++) {
            long rowSum = 0;
            for (int x = 1; x <= width; x++) {
                int pixel = grayscale[y - 1][x - 1];
                rowSum += (long) pixel * pixel;
                squaredIntegral[y][x] = squaredIntegral[y - 1][x] + rowSum;
            }
        }

        return squaredIntegral;
    }

    /**
     * Computes the sum of pixels in a rectangle using the integral image.
     *
     * <p>Uses the formula: sum = ii(D) - ii(B) - ii(C) + ii(A)</p>
     * <pre>
     * A ---- B
     * |      |
     * C ---- D
     * </pre>
     *
     * @param integral the integral image
     * @param x left coordinate
     * @param y top coordinate
     * @param width rectangle width
     * @param height rectangle height
     * @return the sum of pixels in the rectangle
     */
    private long getRectangleSum(long[][] integral, int x, int y, int width, int height) {
        // Bounds checking
        int maxY = integral.length - 1;
        int maxX = integral[0].length - 1;

        int x1 = Math.max(0, x);
        int y1 = Math.max(0, y);
        int x2 = Math.min(maxX, x + width);
        int y2 = Math.min(maxY, y + height);

        return integral[y2][x2] - integral[y1][x2] - integral[y2][x1] + integral[y1][x1];
    }

    /**
     * Evaluates the cascade classifier at a specific window position.
     *
     * <p>Computes Haar features and combines them using the weighted sum.
     * Features are normalized by the window variance for illumination invariance.</p>
     *
     * @param integral the integral image
     * @param squaredIntegral the squared integral image
     * @param x window x position
     * @param y window y position
     * @param windowSize current window size
     * @param scale current scale factor
     * @return confidence score (0.0 to 1.0)
     */
    private double evaluateCascade(long[][] integral, long[][] squaredIntegral,
                                   int x, int y, int windowSize, double scale) {

        // Calculate window variance for normalization
        double variance = computeWindowVariance(integral, squaredIntegral, x, y, windowSize);

        // Avoid division by zero
        if (variance < 1.0) {
            variance = 1.0;
        }

        double stdDev = Math.sqrt(variance);
        double totalScore = 0.0;
        double totalWeight = 0.0;
        int passedStages = 0;

        // Evaluate each Haar feature
        for (int i = 0; i < featureTemplates.size(); i++) {
            HaarFeature feature = featureTemplates.get(i);

            // Scale feature coordinates to current window size
            double featureScale = windowSize / 24.0; // 24x24 is base window
            int scaledX = x + (int) (feature.x * featureScale);
            int scaledY = y + (int) (feature.y * featureScale);
            int scaledWidth = Math.max(2, (int) (feature.width * featureScale));
            int scaledHeight = Math.max(2, (int) (feature.height * featureScale));

            // Compute feature value
            double featureValue = computeHaarFeatureValue(
                integral, scaledX, scaledY, scaledWidth, scaledHeight, feature.type
            );

            // Normalize by variance
            double normalizedValue = featureValue / (stdDev * windowSize * windowSize);

            // Apply weak classifier
            double classifierOutput = (normalizedValue > classifierThresholds[i]) ? 1.0 : -1.0;

            totalScore += classifierWeights[i] * classifierOutput;
            totalWeight += classifierWeights[i];

            // Early rejection (simplified cascade)
            if (i % 5 == 4) { // Check every 5 features
                if (totalScore < -totalWeight * 0.3) {
                    return 0.0; // Reject window early
                }
                passedStages++;
            }
        }

        // Convert score to confidence (sigmoid normalization)
        double normalizedScore = totalScore / totalWeight;
        double confidence = 1.0 / (1.0 + Math.exp(-3.0 * normalizedScore));

        // Boost confidence if all stages passed
        if (passedStages >= featureTemplates.size() / 5) {
            confidence = Math.min(1.0, confidence * 1.1);
        }

        return confidence;
    }

    /**
     * Computes the variance of pixels in a window.
     *
     * <p>Variance = E[X^2] - E[X]^2</p>
     *
     * @param integral the integral image
     * @param squaredIntegral the squared integral image
     * @param x window x position
     * @param y window y position
     * @param size window size
     * @return the variance
     */
    private double computeWindowVariance(long[][] integral, long[][] squaredIntegral,
                                         int x, int y, int size) {
        long sum = getRectangleSum(integral, x, y, size, size);
        long squaredSum = getRectangleSum(squaredIntegral, x, y, size, size);

        double area = size * size;
        double mean = sum / area;
        double meanSquared = squaredSum / area;

        return meanSquared - mean * mean;
    }

    /**
     * Computes the value of a specific Haar-like feature.
     *
     * @param integral the integral image
     * @param x feature x position
     * @param y feature y position
     * @param width feature width
     * @param height feature height
     * @param type the feature type
     * @return the feature value (difference between regions)
     */
    private double computeHaarFeatureValue(long[][] integral, int x, int y,
                                           int width, int height, HaarFeatureType type) {
        switch (type) {
            case TWO_RECT_HORIZONTAL:
                return computeTwoRectHorizontal(integral, x, y, width, height);
            case TWO_RECT_VERTICAL:
                return computeTwoRectVertical(integral, x, y, width, height);
            case THREE_RECT_HORIZONTAL:
                return computeThreeRectHorizontal(integral, x, y, width, height);
            case THREE_RECT_VERTICAL:
                return computeThreeRectVertical(integral, x, y, width, height);
            case FOUR_RECT_CHECKER:
                return computeFourRectChecker(integral, x, y, width, height);
            default:
                return 0.0;
        }
    }

    /**
     * Computes two-rectangle horizontal feature.
     * <pre>
     * +-------+-------+
     * | white | black |
     * +-------+-------+
     * </pre>
     */
    private double computeTwoRectHorizontal(long[][] integral, int x, int y, int width, int height) {
        int halfHeight = height / 2;
        long topSum = getRectangleSum(integral, x, y, width, halfHeight);
        long bottomSum = getRectangleSum(integral, x, y + halfHeight, width, halfHeight);
        return topSum - bottomSum;
    }

    /**
     * Computes two-rectangle vertical feature.
     * <pre>
     * +-----+-----+
     * |     |     |
     * |white|black|
     * |     |     |
     * +-----+-----+
     * </pre>
     */
    private double computeTwoRectVertical(long[][] integral, int x, int y, int width, int height) {
        int halfWidth = width / 2;
        long leftSum = getRectangleSum(integral, x, y, halfWidth, height);
        long rightSum = getRectangleSum(integral, x + halfWidth, y, halfWidth, height);
        return leftSum - rightSum;
    }

    /**
     * Computes three-rectangle horizontal feature.
     * <pre>
     * +---+---+---+
     * | w | b | w |
     * +---+---+---+
     * </pre>
     */
    private double computeThreeRectHorizontal(long[][] integral, int x, int y, int width, int height) {
        int thirdHeight = height / 3;
        long topSum = getRectangleSum(integral, x, y, width, thirdHeight);
        long midSum = getRectangleSum(integral, x, y + thirdHeight, width, thirdHeight);
        long bottomSum = getRectangleSum(integral, x, y + 2 * thirdHeight, width, thirdHeight);
        return topSum - 2 * midSum + bottomSum;
    }

    /**
     * Computes three-rectangle vertical feature.
     * <pre>
     * +---+---+---+
     * |   |   |   |
     * | w | b | w |
     * |   |   |   |
     * +---+---+---+
     * </pre>
     */
    private double computeThreeRectVertical(long[][] integral, int x, int y, int width, int height) {
        int thirdWidth = width / 3;
        long leftSum = getRectangleSum(integral, x, y, thirdWidth, height);
        long midSum = getRectangleSum(integral, x + thirdWidth, y, thirdWidth, height);
        long rightSum = getRectangleSum(integral, x + 2 * thirdWidth, y, thirdWidth, height);
        return leftSum - 2 * midSum + rightSum;
    }

    /**
     * Computes four-rectangle (checker) feature.
     * <pre>
     * +---+---+
     * | w | b |
     * +---+---+
     * | b | w |
     * +---+---+
     * </pre>
     */
    private double computeFourRectChecker(long[][] integral, int x, int y, int width, int height) {
        int halfWidth = width / 2;
        int halfHeight = height / 2;

        long topLeft = getRectangleSum(integral, x, y, halfWidth, halfHeight);
        long topRight = getRectangleSum(integral, x + halfWidth, y, halfWidth, halfHeight);
        long bottomLeft = getRectangleSum(integral, x, y + halfHeight, halfWidth, halfHeight);
        long bottomRight = getRectangleSum(integral, x + halfWidth, y + halfHeight, halfWidth, halfHeight);

        return (topLeft + bottomRight) - (topRight + bottomLeft);
    }

    /**
     * Merges overlapping detections using Non-Maximum Suppression (NMS).
     *
     * <p>Detections with IoU (Intersection over Union) above the threshold
     * are merged, keeping the one with highest confidence.</p>
     *
     * @param candidates list of candidate detections
     * @return merged face regions
     */
    private List<FaceRegion> mergeDetections(List<Detection> candidates) {
        if (candidates.isEmpty()) {
            return Collections.emptyList();
        }

        // Sort by confidence descending
        candidates.sort((a, b) -> Double.compare(b.confidence, a.confidence));

        List<FaceRegion> results = new ArrayList<>();
        boolean[] suppressed = new boolean[candidates.size()];

        for (int i = 0; i < candidates.size(); i++) {
            if (suppressed[i]) {
                continue;
            }

            Detection current = candidates.get(i);

            // Count overlapping detections (neighbors)
            int neighborCount = 1;
            double sumX = current.x;
            double sumY = current.y;
            double sumWidth = current.width;
            double sumHeight = current.height;
            double maxConfidence = current.confidence;

            for (int j = i + 1; j < candidates.size(); j++) {
                if (suppressed[j]) {
                    continue;
                }

                Detection other = candidates.get(j);
                double iou = computeIoU(current, other);

                if (iou > NMS_IOU_THRESHOLD) {
                    suppressed[j] = true;
                    neighborCount++;
                    sumX += other.x;
                    sumY += other.y;
                    sumWidth += other.width;
                    sumHeight += other.height;
                    maxConfidence = Math.max(maxConfidence, other.confidence);
                }
            }

            // Only accept if enough neighbors
            if (neighborCount >= minNeighbors) {
                // Average the merged detections
                int avgX = (int) (sumX / neighborCount);
                int avgY = (int) (sumY / neighborCount);
                int avgWidth = (int) (sumWidth / neighborCount);
                int avgHeight = (int) (sumHeight / neighborCount);

                // Boost confidence based on neighbor count
                double finalConfidence = Math.min(1.0,
                    maxConfidence * (1.0 + 0.1 * Math.min(neighborCount - minNeighbors, 5)));

                results.add(new FaceRegion(avgX, avgY, avgWidth, avgHeight, finalConfidence));
            }
        }

        return results;
    }

    /**
     * Computes Intersection over Union (IoU) between two detections.
     *
     * @param a first detection
     * @param b second detection
     * @return IoU value between 0.0 and 1.0
     */
    private double computeIoU(Detection a, Detection b) {
        int x1 = Math.max(a.x, b.x);
        int y1 = Math.max(a.y, b.y);
        int x2 = Math.min(a.x + a.width, b.x + b.width);
        int y2 = Math.min(a.y + a.height, b.y + b.height);

        if (x1 >= x2 || y1 >= y2) {
            return 0.0;
        }

        double intersection = (x2 - x1) * (y2 - y1);
        double areaA = a.width * a.height;
        double areaB = b.width * b.height;
        double union = areaA + areaB - intersection;

        return intersection / union;
    }

    /**
     * {@inheritDoc}
     *
     * <p>This implementation does not support landmark detection.
     * Use a dedicated landmark detector after face detection.</p>
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
     * Gets the scale factor used for multi-scale detection.
     *
     * @return the scale factor
     */
    public double getScaleFactor() {
        return scaleFactor;
    }

    /**
     * Sets the scale factor for multi-scale detection.
     *
     * @param scaleFactor the scale factor (must be > 1.0)
     */
    public void setScaleFactor(double scaleFactor) {
        if (scaleFactor <= 1.0 || scaleFactor > 2.0) {
            throw new IllegalArgumentException("Scale factor must be between 1.0 (exclusive) and 2.0");
        }
        this.scaleFactor = scaleFactor;
    }

    /**
     * Gets the minimum neighbor count for detection acceptance.
     *
     * @return the minimum neighbors
     */
    public int getMinNeighbors() {
        return minNeighbors;
    }

    /**
     * Sets the minimum neighbor count for detection acceptance.
     *
     * @param minNeighbors the minimum neighbors (0 or more)
     */
    public void setMinNeighbors(int minNeighbors) {
        if (minNeighbors < 0) {
            throw new IllegalArgumentException("Minimum neighbors cannot be negative");
        }
        this.minNeighbors = minNeighbors;
    }

    /**
     * Gets the detection threshold.
     *
     * @return the detection threshold (0.0 to 1.0)
     */
    public double getDetectionThreshold() {
        return detectionThreshold;
    }

    /**
     * Sets the detection threshold.
     *
     * @param threshold the threshold (0.0 to 1.0)
     */
    public void setDetectionThreshold(double threshold) {
        if (threshold < 0.0 || threshold > 1.0) {
            throw new IllegalArgumentException("Threshold must be between 0.0 and 1.0");
        }
        this.detectionThreshold = threshold;
    }

    @Override
    public String toString() {
        return String.format("ViolaJonesFaceDetector{minSize=%d, scale=%.2f, minNeighbors=%d, threshold=%.2f}",
            minFaceSize, scaleFactor, minNeighbors, detectionThreshold);
    }

    // =========================================================================
    // Inner Classes
    // =========================================================================

    /**
     * Types of Haar-like features.
     */
    private enum HaarFeatureType {
        /** Two rectangles side by side horizontally. */
        TWO_RECT_HORIZONTAL,
        /** Two rectangles stacked vertically. */
        TWO_RECT_VERTICAL,
        /** Three rectangles in a horizontal row. */
        THREE_RECT_HORIZONTAL,
        /** Three rectangles in a vertical column. */
        THREE_RECT_VERTICAL,
        /** Four rectangles in a checker pattern. */
        FOUR_RECT_CHECKER
    }

    /**
     * Represents a Haar-like feature template.
     */
    private static class HaarFeature implements Serializable {
        private static final long serialVersionUID = 1L;

        final HaarFeatureType type;
        final int x;
        final int y;
        final int width;
        final int height;

        HaarFeature(HaarFeatureType type, int x, int y, int width, int height) {
            this.type = type;
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
        }
    }

    /**
     * Represents a candidate detection before merging.
     */
    private static class Detection {
        final int x;
        final int y;
        final int width;
        final int height;
        final double confidence;

        Detection(int x, int y, int width, int height, double confidence) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.confidence = confidence;
        }
    }
}
