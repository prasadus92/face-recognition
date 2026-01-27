package com.facerecognition.infrastructure.detection;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FaceLandmarks;
import com.facerecognition.domain.model.FaceRegion;
import com.facerecognition.domain.service.FaceDetector;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.*;
import java.util.stream.Collectors;

/**
 * Composite face detector that combines multiple detection methods using ensemble voting.
 *
 * <p>This detector aggregates results from multiple underlying detectors to improve
 * accuracy and robustness. It supports various combination strategies:</p>
 * <ul>
 *   <li><b>Voting (Union)</b>: Accept detections confirmed by minimum number of detectors</li>
 *   <li><b>Intersection</b>: Only accept detections found by all detectors</li>
 *   <li><b>Weighted</b>: Combine detections using detector-specific weights</li>
 *   <li><b>Cascade</b>: Use detectors in sequence, stopping on first success</li>
 *   <li><b>Fallback</b>: Try detectors in order until one succeeds</li>
 * </ul>
 *
 * <h3>Algorithm Overview:</h3>
 * <ol>
 *   <li>Run all registered detectors (optionally in parallel)</li>
 *   <li>Collect all detection results with their confidence scores</li>
 *   <li>Group overlapping detections from different detectors</li>
 *   <li>Apply voting/ensemble strategy to filter results</li>
 *   <li>Merge final detections with combined confidence</li>
 * </ol>
 *
 * <h3>Voting Strategy:</h3>
 * <pre>
 * For each candidate region:
 *   votes = count of detectors that found overlapping detection
 *   if votes >= minVotes:
 *     combined_confidence = weighted_average(individual_confidences)
 *     accept detection
 * </pre>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * // Create composite detector with multiple methods
 * CompositeFaceDetector composite = new CompositeFaceDetector();
 * composite.addDetector(new ViolaJonesFaceDetector(), 1.0);
 * composite.addDetector(new SkinColorDetector(), 0.7);
 *
 * // Configure ensemble strategy
 * composite.setStrategy(EnsembleStrategy.VOTING);
 * composite.setMinVotes(2);
 *
 * // Detect faces
 * List<FaceRegion> faces = composite.detectFaces(image);
 * }</pre>
 *
 * <h3>Thread Safety:</h3>
 * <p>This class is thread-safe when using parallel detection. Individual
 * detector implementations must also be thread-safe for parallel mode.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see FaceDetector
 * @see ViolaJonesFaceDetector
 * @see SkinColorDetector
 */
public class CompositeFaceDetector implements FaceDetector, Serializable {

    private static final long serialVersionUID = 1L;

    /** Algorithm name for identification. */
    public static final String ALGORITHM_NAME = "Composite";

    /** Current algorithm version. */
    public static final String VERSION = "2.0";

    /** Default IoU threshold for matching detections across detectors. */
    public static final double DEFAULT_IOU_THRESHOLD = 0.4;

    /** Default minimum votes required for voting strategy. */
    public static final int DEFAULT_MIN_VOTES = 1;

    /** Default timeout for parallel detection in milliseconds. */
    public static final long DEFAULT_TIMEOUT_MS = 10000;

    /**
     * Ensemble strategies for combining detector results.
     */
    public enum EnsembleStrategy {
        /**
         * Accept detections with minimum vote count.
         * More votes increase confidence.
         */
        VOTING,

        /**
         * Only accept detections found by all detectors.
         * Most conservative approach.
         */
        INTERSECTION,

        /**
         * Combine all detections with weighted confidence.
         * Most inclusive approach.
         */
        UNION_WEIGHTED,

        /**
         * Try detectors in sequence, use first successful result.
         * Fastest when first detector succeeds.
         */
        CASCADE,

        /**
         * Try detectors in order, use first that returns results.
         * Good for detector fallback chains.
         */
        FALLBACK
    }

    // Registered detectors with their weights
    private final List<DetectorEntry> detectors;

    // Configuration
    private EnsembleStrategy strategy;
    private int minVotes;
    private double iouThreshold;
    private int minFaceSize;
    private boolean parallelExecution;
    private long timeoutMs;
    private ExecutorService executorService;

    /**
     * Creates a composite detector with default settings.
     *
     * <p>Default configuration:</p>
     * <ul>
     *   <li>Strategy: VOTING</li>
     *   <li>Minimum votes: 1</li>
     *   <li>IoU threshold: 0.4</li>
     *   <li>Parallel execution: disabled</li>
     * </ul>
     */
    public CompositeFaceDetector() {
        this.detectors = new ArrayList<>();
        this.strategy = EnsembleStrategy.VOTING;
        this.minVotes = DEFAULT_MIN_VOTES;
        this.iouThreshold = DEFAULT_IOU_THRESHOLD;
        this.minFaceSize = 30;
        this.parallelExecution = false;
        this.timeoutMs = DEFAULT_TIMEOUT_MS;
        this.executorService = null;
    }

    /**
     * Creates a composite detector with specified detectors.
     *
     * @param detectors map of detectors to their weights
     */
    public CompositeFaceDetector(Map<FaceDetector, Double> detectors) {
        this();
        for (Map.Entry<FaceDetector, Double> entry : detectors.entrySet()) {
            addDetector(entry.getKey(), entry.getValue());
        }
    }

    /**
     * Adds a detector with default weight of 1.0.
     *
     * @param detector the detector to add
     * @return this instance for method chaining
     * @throws IllegalArgumentException if detector is null or already registered
     */
    public CompositeFaceDetector addDetector(FaceDetector detector) {
        return addDetector(detector, 1.0);
    }

    /**
     * Adds a detector with specified weight.
     *
     * <p>Weight affects the contribution of this detector's confidence
     * scores to the final combined confidence. Higher weights have more
     * influence on the final score.</p>
     *
     * @param detector the detector to add
     * @param weight the detector weight (must be positive)
     * @return this instance for method chaining
     * @throws IllegalArgumentException if detector is null, weight is invalid,
     *         or detector is already registered
     */
    public CompositeFaceDetector addDetector(FaceDetector detector, double weight) {
        Objects.requireNonNull(detector, "Detector cannot be null");
        if (weight <= 0) {
            throw new IllegalArgumentException("Weight must be positive");
        }
        if (detector == this) {
            throw new IllegalArgumentException("Cannot add self as detector");
        }

        // Check for duplicate
        for (DetectorEntry entry : detectors) {
            if (entry.detector == detector) {
                throw new IllegalArgumentException("Detector already registered: " + detector.getName());
            }
        }

        detectors.add(new DetectorEntry(detector, weight));
        return this;
    }

    /**
     * Removes a detector from the composite.
     *
     * @param detector the detector to remove
     * @return true if detector was removed
     */
    public boolean removeDetector(FaceDetector detector) {
        return detectors.removeIf(entry -> entry.detector == detector);
    }

    /**
     * Gets the list of registered detectors.
     *
     * @return unmodifiable list of detector entries
     */
    public List<FaceDetector> getDetectors() {
        return detectors.stream()
            .map(e -> e.detector)
            .collect(Collectors.toUnmodifiableList());
    }

    /**
     * Gets the weight assigned to a specific detector.
     *
     * @param detector the detector
     * @return the weight, or -1 if not found
     */
    public double getDetectorWeight(FaceDetector detector) {
        for (DetectorEntry entry : detectors) {
            if (entry.detector == detector) {
                return entry.weight;
            }
        }
        return -1;
    }

    /**
     * Sets the weight for an existing detector.
     *
     * @param detector the detector
     * @param weight the new weight
     * @throws IllegalArgumentException if detector not found or weight invalid
     */
    public void setDetectorWeight(FaceDetector detector, double weight) {
        if (weight <= 0) {
            throw new IllegalArgumentException("Weight must be positive");
        }
        for (DetectorEntry entry : detectors) {
            if (entry.detector == detector) {
                entry.weight = weight;
                return;
            }
        }
        throw new IllegalArgumentException("Detector not found: " + detector.getName());
    }

    /**
     * {@inheritDoc}
     *
     * <p>Runs all registered detectors and combines their results
     * according to the configured ensemble strategy.</p>
     */
    @Override
    public List<FaceRegion> detectFaces(FaceImage image) {
        return detectFaces(image, 0.0);
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

        if (detectors.isEmpty()) {
            return Collections.emptyList();
        }

        // Execute detection based on strategy
        List<FaceRegion> results;
        switch (strategy) {
            case CASCADE:
                results = detectCascade(image, minConfidence);
                break;
            case FALLBACK:
                results = detectFallback(image, minConfidence);
                break;
            default:
                results = detectEnsemble(image, minConfidence);
                break;
        }

        // Sort by confidence
        results.sort((a, b) -> Double.compare(b.getConfidence(), a.getConfidence()));

        return results;
    }

    /**
     * Executes ensemble detection (VOTING, INTERSECTION, UNION_WEIGHTED).
     */
    private List<FaceRegion> detectEnsemble(FaceImage image, double minConfidence) {
        // Collect results from all detectors
        List<DetectorResult> allResults = parallelExecution ?
            runDetectorsParallel(image, minConfidence) :
            runDetectorsSequential(image, minConfidence);

        if (allResults.isEmpty()) {
            return Collections.emptyList();
        }

        // Group overlapping detections
        List<DetectionGroup> groups = groupOverlappingDetections(allResults);

        // Apply ensemble strategy
        return applyEnsembleStrategy(groups, minConfidence);
    }

    /**
     * Runs all detectors sequentially.
     */
    private List<DetectorResult> runDetectorsSequential(FaceImage image, double minConfidence) {
        List<DetectorResult> results = new ArrayList<>();

        for (DetectorEntry entry : detectors) {
            try {
                List<FaceRegion> detections = entry.detector.detectFaces(image, minConfidence);
                for (FaceRegion face : detections) {
                    results.add(new DetectorResult(face, entry.detector, entry.weight));
                }
            } catch (Exception e) {
                // Log error but continue with other detectors
                System.err.println("Detector " + entry.detector.getName() + " failed: " + e.getMessage());
            }
        }

        return results;
    }

    /**
     * Runs all detectors in parallel using thread pool.
     */
    private List<DetectorResult> runDetectorsParallel(FaceImage image, double minConfidence) {
        ExecutorService executor = executorService != null ?
            executorService :
            Executors.newFixedThreadPool(Math.min(detectors.size(), 4));

        try {
            List<Future<List<DetectorResult>>> futures = new ArrayList<>();

            for (DetectorEntry entry : detectors) {
                futures.add(executor.submit(() -> {
                    List<DetectorResult> results = new ArrayList<>();
                    try {
                        List<FaceRegion> detections = entry.detector.detectFaces(image, minConfidence);
                        for (FaceRegion face : detections) {
                            results.add(new DetectorResult(face, entry.detector, entry.weight));
                        }
                    } catch (Exception e) {
                        System.err.println("Detector " + entry.detector.getName() + " failed: " + e.getMessage());
                    }
                    return results;
                }));
            }

            // Collect results with timeout
            List<DetectorResult> allResults = new ArrayList<>();
            for (Future<List<DetectorResult>> future : futures) {
                try {
                    allResults.addAll(future.get(timeoutMs, TimeUnit.MILLISECONDS));
                } catch (TimeoutException e) {
                    future.cancel(true);
                    System.err.println("Detector timed out");
                } catch (Exception e) {
                    System.err.println("Detector execution failed: " + e.getMessage());
                }
            }

            return allResults;

        } finally {
            if (executorService == null) {
                executor.shutdown();
            }
        }
    }

    /**
     * Groups overlapping detections from different detectors.
     *
     * <p>Detections are considered overlapping if their IoU exceeds
     * the configured threshold.</p>
     */
    private List<DetectionGroup> groupOverlappingDetections(List<DetectorResult> results) {
        List<DetectionGroup> groups = new ArrayList<>();
        boolean[] assigned = new boolean[results.size()];

        for (int i = 0; i < results.size(); i++) {
            if (assigned[i]) {
                continue;
            }

            DetectionGroup group = new DetectionGroup();
            group.addResult(results.get(i));
            assigned[i] = true;

            // Find overlapping detections
            for (int j = i + 1; j < results.size(); j++) {
                if (assigned[j]) {
                    continue;
                }

                // Check if this detection overlaps with any in the group
                FaceRegion candidate = results.get(j).region;
                boolean overlaps = false;

                for (DetectorResult existing : group.results) {
                    double iou = existing.region.intersectionOverUnion(candidate);
                    if (iou >= iouThreshold) {
                        overlaps = true;
                        break;
                    }
                }

                if (overlaps) {
                    group.addResult(results.get(j));
                    assigned[j] = true;
                }
            }

            groups.add(group);
        }

        return groups;
    }

    /**
     * Applies the configured ensemble strategy to detection groups.
     */
    private List<FaceRegion> applyEnsembleStrategy(List<DetectionGroup> groups,
                                                    double minConfidence) {
        List<FaceRegion> results = new ArrayList<>();

        for (DetectionGroup group : groups) {
            FaceRegion merged = null;

            switch (strategy) {
                case VOTING:
                    if (group.getVoteCount() >= minVotes) {
                        merged = mergeGroupVoting(group);
                    }
                    break;

                case INTERSECTION:
                    // Require detection from all detectors
                    if (group.getUniqueDetectorCount() == detectors.size()) {
                        merged = mergeGroupVoting(group);
                    }
                    break;

                case UNION_WEIGHTED:
                    // Accept all groups, weight by number of detectors
                    merged = mergeGroupWeighted(group);
                    break;

                default:
                    merged = mergeGroupVoting(group);
            }

            if (merged != null && merged.getConfidence() >= minConfidence) {
                results.add(merged);
            }
        }

        return results;
    }

    /**
     * Merges a detection group using voting/averaging.
     */
    private FaceRegion mergeGroupVoting(DetectionGroup group) {
        if (group.results.isEmpty()) {
            return null;
        }

        // Calculate weighted average of bounding boxes
        double sumX = 0, sumY = 0, sumWidth = 0, sumHeight = 0;
        double totalWeight = 0;
        double maxConfidence = 0;

        for (DetectorResult result : group.results) {
            FaceRegion r = result.region;
            double w = result.weight;

            sumX += r.getX() * w;
            sumY += r.getY() * w;
            sumWidth += r.getWidth() * w;
            sumHeight += r.getHeight() * w;
            totalWeight += w;
            maxConfidence = Math.max(maxConfidence, r.getConfidence());
        }

        int avgX = (int) (sumX / totalWeight);
        int avgY = (int) (sumY / totalWeight);
        int avgWidth = (int) (sumWidth / totalWeight);
        int avgHeight = (int) (sumHeight / totalWeight);

        // Boost confidence based on vote count
        int votes = group.getVoteCount();
        double voteBoost = 1.0 + 0.1 * Math.min(votes - 1, 5);
        double combinedConfidence = Math.min(1.0, maxConfidence * voteBoost);

        // Ensure minimum dimensions
        avgWidth = Math.max(avgWidth, minFaceSize);
        avgHeight = Math.max(avgHeight, minFaceSize);

        return new FaceRegion(avgX, avgY, avgWidth, avgHeight, combinedConfidence);
    }

    /**
     * Merges a detection group using weighted combination.
     */
    private FaceRegion mergeGroupWeighted(DetectionGroup group) {
        if (group.results.isEmpty()) {
            return null;
        }

        // Calculate weighted average
        double sumX = 0, sumY = 0, sumWidth = 0, sumHeight = 0;
        double sumConfidence = 0;
        double totalWeight = 0;

        for (DetectorResult result : group.results) {
            FaceRegion r = result.region;
            double w = result.weight * r.getConfidence(); // Weight by both detector weight and confidence

            sumX += r.getX() * w;
            sumY += r.getY() * w;
            sumWidth += r.getWidth() * w;
            sumHeight += r.getHeight() * w;
            sumConfidence += r.getConfidence() * result.weight;
            totalWeight += w;
        }

        // Normalize by total detector weights for confidence
        double detectorWeightSum = group.results.stream()
            .mapToDouble(r -> r.weight)
            .sum();

        int avgX = (int) (sumX / totalWeight);
        int avgY = (int) (sumY / totalWeight);
        int avgWidth = Math.max((int) (sumWidth / totalWeight), minFaceSize);
        int avgHeight = Math.max((int) (sumHeight / totalWeight), minFaceSize);
        double avgConfidence = sumConfidence / detectorWeightSum;

        return new FaceRegion(avgX, avgY, avgWidth, avgHeight, Math.min(1.0, avgConfidence));
    }

    /**
     * Executes cascade detection - tries detectors in order, stops on first hit.
     */
    private List<FaceRegion> detectCascade(FaceImage image, double minConfidence) {
        for (DetectorEntry entry : detectors) {
            try {
                List<FaceRegion> results = entry.detector.detectFaces(image, minConfidence);
                if (!results.isEmpty()) {
                    // Weight the confidence scores
                    return results.stream()
                        .map(r -> new FaceRegion(
                            r.getX(), r.getY(), r.getWidth(), r.getHeight(),
                            Math.min(1.0, r.getConfidence() * entry.weight)
                        ))
                        .collect(Collectors.toList());
                }
            } catch (Exception e) {
                System.err.println("Detector " + entry.detector.getName() + " failed: " + e.getMessage());
            }
        }
        return Collections.emptyList();
    }

    /**
     * Executes fallback detection - returns first non-empty result.
     */
    private List<FaceRegion> detectFallback(FaceImage image, double minConfidence) {
        for (DetectorEntry entry : detectors) {
            try {
                List<FaceRegion> results = entry.detector.detectFaces(image, minConfidence);
                if (!results.isEmpty()) {
                    return new ArrayList<>(results);
                }
            } catch (Exception e) {
                // Continue to next detector
                System.err.println("Detector " + entry.detector.getName() + " failed, trying fallback");
            }
        }
        return Collections.emptyList();
    }

    /**
     * {@inheritDoc}
     *
     * <p>Returns true if any registered detector supports landmarks.</p>
     */
    @Override
    public boolean supportsLandmarks() {
        return detectors.stream().anyMatch(e -> e.detector.supportsLandmarks());
    }

    /**
     * {@inheritDoc}
     *
     * <p>Attempts landmark detection using the first supporting detector.</p>
     */
    @Override
    public Optional<FaceLandmarks> detectLandmarks(FaceImage image, FaceRegion faceRegion) {
        for (DetectorEntry entry : detectors) {
            if (entry.detector.supportsLandmarks()) {
                Optional<FaceLandmarks> landmarks = entry.detector.detectLandmarks(image, faceRegion);
                if (landmarks.isPresent()) {
                    return landmarks;
                }
            }
        }
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
     *
     * <p>Also propagates the setting to all registered detectors.</p>
     */
    @Override
    public void setMinFaceSize(int minSize) {
        if (minSize < 10) {
            throw new IllegalArgumentException("Minimum face size must be at least 10 pixels");
        }
        this.minFaceSize = minSize;

        // Propagate to all detectors
        for (DetectorEntry entry : detectors) {
            try {
                entry.detector.setMinFaceSize(minSize);
            } catch (Exception e) {
                // Some detectors might have different constraints
            }
        }
    }

    /**
     * Gets the current ensemble strategy.
     *
     * @return the ensemble strategy
     */
    public EnsembleStrategy getStrategy() {
        return strategy;
    }

    /**
     * Sets the ensemble strategy.
     *
     * @param strategy the strategy to use
     * @return this instance for method chaining
     */
    public CompositeFaceDetector setStrategy(EnsembleStrategy strategy) {
        this.strategy = Objects.requireNonNull(strategy, "Strategy cannot be null");
        return this;
    }

    /**
     * Gets the minimum vote count for VOTING strategy.
     *
     * @return the minimum votes required
     */
    public int getMinVotes() {
        return minVotes;
    }

    /**
     * Sets the minimum vote count for VOTING strategy.
     *
     * @param minVotes minimum votes (must be positive)
     * @return this instance for method chaining
     */
    public CompositeFaceDetector setMinVotes(int minVotes) {
        if (minVotes < 1) {
            throw new IllegalArgumentException("Minimum votes must be at least 1");
        }
        this.minVotes = minVotes;
        return this;
    }

    /**
     * Gets the IoU threshold for matching detections.
     *
     * @return the IoU threshold
     */
    public double getIouThreshold() {
        return iouThreshold;
    }

    /**
     * Sets the IoU threshold for matching detections across detectors.
     *
     * @param threshold IoU threshold (0.0 to 1.0)
     * @return this instance for method chaining
     */
    public CompositeFaceDetector setIouThreshold(double threshold) {
        if (threshold < 0.0 || threshold > 1.0) {
            throw new IllegalArgumentException("IoU threshold must be between 0.0 and 1.0");
        }
        this.iouThreshold = threshold;
        return this;
    }

    /**
     * Checks if parallel execution is enabled.
     *
     * @return true if parallel execution is enabled
     */
    public boolean isParallelExecution() {
        return parallelExecution;
    }

    /**
     * Enables or disables parallel detector execution.
     *
     * @param parallel true to enable parallel execution
     * @return this instance for method chaining
     */
    public CompositeFaceDetector setParallelExecution(boolean parallel) {
        this.parallelExecution = parallel;
        return this;
    }

    /**
     * Gets the timeout for parallel execution.
     *
     * @return timeout in milliseconds
     */
    public long getTimeoutMs() {
        return timeoutMs;
    }

    /**
     * Sets the timeout for parallel execution.
     *
     * @param timeoutMs timeout in milliseconds
     * @return this instance for method chaining
     */
    public CompositeFaceDetector setTimeoutMs(long timeoutMs) {
        if (timeoutMs <= 0) {
            throw new IllegalArgumentException("Timeout must be positive");
        }
        this.timeoutMs = timeoutMs;
        return this;
    }

    /**
     * Sets a custom executor service for parallel detection.
     *
     * @param executor the executor service (null to use default)
     */
    public void setExecutorService(ExecutorService executor) {
        this.executorService = executor;
    }

    /**
     * Gets the number of registered detectors.
     *
     * @return detector count
     */
    public int getDetectorCount() {
        return detectors.size();
    }

    /**
     * Clears all registered detectors.
     */
    public void clearDetectors() {
        detectors.clear();
    }

    /**
     * Creates a standard composite detector with ViolaJones and SkinColor.
     *
     * <p>Configures:</p>
     * <ul>
     *   <li>ViolaJonesFaceDetector with weight 1.0</li>
     *   <li>SkinColorDetector with weight 0.7</li>
     *   <li>VOTING strategy with minVotes=1</li>
     * </ul>
     *
     * @return configured composite detector
     */
    public static CompositeFaceDetector createDefault() {
        CompositeFaceDetector composite = new CompositeFaceDetector();
        composite.addDetector(new ViolaJonesFaceDetector(), 1.0);
        composite.addDetector(new SkinColorDetector(), 0.7);
        composite.setStrategy(EnsembleStrategy.VOTING);
        composite.setMinVotes(1);
        return composite;
    }

    /**
     * Creates a high-accuracy composite detector.
     *
     * <p>Configures:</p>
     * <ul>
     *   <li>ViolaJonesFaceDetector with weight 1.0</li>
     *   <li>SkinColorDetector with weight 0.5</li>
     *   <li>VOTING strategy with minVotes=2</li>
     * </ul>
     *
     * @return configured composite detector
     */
    public static CompositeFaceDetector createHighAccuracy() {
        CompositeFaceDetector composite = new CompositeFaceDetector();
        composite.addDetector(new ViolaJonesFaceDetector(), 1.0);
        composite.addDetector(new SkinColorDetector(), 0.5);
        composite.setStrategy(EnsembleStrategy.VOTING);
        composite.setMinVotes(2);
        return composite;
    }

    /**
     * Creates a fast composite detector with fallback.
     *
     * <p>Configures:</p>
     * <ul>
     *   <li>ViolaJonesFaceDetector (primary)</li>
     *   <li>SkinColorDetector (fallback)</li>
     *   <li>FALLBACK strategy</li>
     * </ul>
     *
     * @return configured composite detector
     */
    public static CompositeFaceDetector createFastFallback() {
        CompositeFaceDetector composite = new CompositeFaceDetector();
        composite.addDetector(new ViolaJonesFaceDetector(), 1.0);
        composite.addDetector(new SkinColorDetector(), 0.8);
        composite.setStrategy(EnsembleStrategy.FALLBACK);
        return composite;
    }

    @Override
    public String toString() {
        String detectorNames = detectors.stream()
            .map(e -> e.detector.getName())
            .collect(Collectors.joining(", "));
        return String.format("CompositeFaceDetector{strategy=%s, detectors=[%s], minVotes=%d}",
            strategy, detectorNames, minVotes);
    }

    // =========================================================================
    // Inner Classes
    // =========================================================================

    /**
     * Entry holding a detector and its weight.
     */
    private static class DetectorEntry implements Serializable {
        private static final long serialVersionUID = 1L;

        final FaceDetector detector;
        double weight;

        DetectorEntry(FaceDetector detector, double weight) {
            this.detector = detector;
            this.weight = weight;
        }
    }

    /**
     * Result from a single detector.
     */
    private static class DetectorResult {
        final FaceRegion region;
        final FaceDetector detector;
        final double weight;

        DetectorResult(FaceRegion region, FaceDetector detector, double weight) {
            this.region = region;
            this.detector = detector;
            this.weight = weight;
        }
    }

    /**
     * Group of overlapping detections from potentially multiple detectors.
     */
    private static class DetectionGroup {
        final List<DetectorResult> results = new ArrayList<>();
        final Set<FaceDetector> detectors = new HashSet<>();

        void addResult(DetectorResult result) {
            results.add(result);
            detectors.add(result.detector);
        }

        int getVoteCount() {
            return detectors.size();
        }

        int getUniqueDetectorCount() {
            return detectors.size();
        }
    }
}
