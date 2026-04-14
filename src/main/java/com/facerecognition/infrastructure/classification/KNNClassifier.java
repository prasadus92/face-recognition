package com.facerecognition.infrastructure.classification;

import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.domain.service.FaceClassifier;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * K-Nearest Neighbors classifier for face recognition.
 *
 * <p>This classifier finds the K closest enrolled samples to a probe
 * face and determines identity based on voting or distance weighting.</p>
 *
 * <h3>Classification Modes:</h3>
 * <ul>
 *   <li><b>k=1</b>: Simple nearest neighbor</li>
 *   <li><b>k>1</b>: Majority voting among k nearest neighbors</li>
 *   <li><b>Weighted</b>: Distance-weighted voting</li>
 * </ul>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 1.0
 * @see FaceClassifier
 */
public class KNNClassifier implements FaceClassifier, Serializable {

    private static final long serialVersionUID = 2L;

    private static final String NAME = "KNN";

    private final ClassifierConfig config;
    private final Map<String, Identity> enrolledIdentities;
    private DistanceMetric distanceMetric;

    /**
     * Creates a KNN classifier with default settings.
     */
    public KNNClassifier() {
        this(new ClassifierConfig());
    }

    /**
     * Creates a KNN classifier with custom configuration.
     *
     * @param config the classifier configuration
     */
    public KNNClassifier(ClassifierConfig config) {
        this.config = Objects.requireNonNull(config);
        this.enrolledIdentities = new ConcurrentHashMap<>();
        this.distanceMetric = config.getMetric();
    }

    @Override
    public void enroll(Identity identity) {
        Objects.requireNonNull(identity, "Identity cannot be null");
        if (!identity.hasSamples()) {
            throw new IllegalArgumentException("Identity must have at least one enrolled sample");
        }
        enrolledIdentities.put(identity.getId(), identity);
    }

    @Override
    public boolean unenroll(String identityId) {
        return enrolledIdentities.remove(identityId) != null;
    }

    @Override
    public boolean isEnrolled(String identityId) {
        return enrolledIdentities.containsKey(identityId);
    }

    @Override
    public int getEnrolledCount() {
        return enrolledIdentities.size();
    }

    @Override
    public List<Identity> getEnrolledIdentities() {
        return new ArrayList<>(enrolledIdentities.values());
    }

    @Override
    public RecognitionResult classify(FeatureVector probe) {
        return classify(probe, config.getThreshold());
    }

    @Override
    public RecognitionResult classify(FeatureVector probe, double threshold) {
        if (enrolledIdentities.isEmpty()) {
            return RecognitionResult.builder()
                .status(RecognitionResult.Status.UNKNOWN)
                .extractedFeatures(probe)
                .build();
        }

        // Find all distances
        List<DistanceEntry> distances = computeAllDistances(probe);

        if (distances.isEmpty()) {
            return RecognitionResult.builder()
                .status(RecognitionResult.Status.UNKNOWN)
                .extractedFeatures(probe)
                .build();
        }

        // Sort by distance
        distances.sort(Comparator.comparingDouble(e -> e.distance));

        // Get best match
        DistanceEntry best = distances.get(0);
        double confidence = distanceToConfidence(best.distance);

        // Check threshold
        if (confidence < threshold) {
            // Build result with alternatives but no match
            List<RecognitionResult.MatchResult> alternatives = buildAlternatives(distances, config.getK());
            return RecognitionResult.builder()
                .status(RecognitionResult.Status.UNKNOWN)
                .alternatives(alternatives)
                .extractedFeatures(probe)
                .build();
        }

        // Build match result
        RecognitionResult.MatchResult bestMatch = new RecognitionResult.MatchResult(
            best.identity, confidence, best.distance);

        List<RecognitionResult.MatchResult> alternatives = buildAlternatives(distances.subList(1, distances.size()), config.getK() - 1);

        return RecognitionResult.builder()
            .status(RecognitionResult.Status.RECOGNIZED)
            .bestMatch(bestMatch)
            .alternatives(alternatives)
            .extractedFeatures(probe)
            .build();
    }

    @Override
    public RecognitionResult getTopMatches(FeatureVector probe, int n) {
        List<DistanceEntry> distances = computeAllDistances(probe);
        distances.sort(Comparator.comparingDouble(e -> e.distance));

        List<RecognitionResult.MatchResult> alternatives = buildAlternatives(distances, n);

        if (alternatives.isEmpty()) {
            return RecognitionResult.builder()
                .status(RecognitionResult.Status.UNKNOWN)
                .extractedFeatures(probe)
                .build();
        }

        RecognitionResult.MatchResult best = alternatives.get(0);
        return RecognitionResult.builder()
            .status(best.getConfidence() >= config.getThreshold()
                ? RecognitionResult.Status.RECOGNIZED
                : RecognitionResult.Status.UNKNOWN)
            .bestMatch(best)
            .alternatives(alternatives.subList(1, alternatives.size()))
            .extractedFeatures(probe)
            .build();
    }

    @Override
    public double getDistance(FeatureVector probe, String identityId) {
        Identity identity = enrolledIdentities.get(identityId);
        if (identity == null) {
            return Double.MAX_VALUE;
        }
        return computeMinDistance(probe, identity);
    }

    @Override
    public String getName() {
        return NAME;
    }

    @Override
    public DistanceMetric getDistanceMetric() {
        return distanceMetric;
    }

    @Override
    public void setDistanceMetric(DistanceMetric metric) {
        this.distanceMetric = Objects.requireNonNull(metric);
    }

    @Override
    public void clear() {
        enrolledIdentities.clear();
    }

    @Override
    public void retrain() {
        // KNN doesn't require training
    }

    private List<DistanceEntry> computeAllDistances(FeatureVector probe) {
        List<DistanceEntry> entries = new ArrayList<>();

        for (Identity identity : enrolledIdentities.values()) {
            if (!identity.isActive()) continue;

            double distance = config.isUseAverageFeatures()
                ? computeDistanceToAverage(probe, identity)
                : computeMinDistance(probe, identity);

            entries.add(new DistanceEntry(identity, distance));
        }

        return entries;
    }

    private double computeMinDistance(FeatureVector probe, Identity identity) {
        double minDistance = Double.MAX_VALUE;

        for (Identity.EnrolledSample sample : identity.getSamples()) {
            double distance = computeDistance(probe, sample.getFeatures());
            minDistance = Math.min(minDistance, distance);
        }

        return minDistance;
    }

    private double computeDistanceToAverage(FeatureVector probe, Identity identity) {
        FeatureVector average = identity.getAverageFeatureVector();
        if (average == null) {
            return Double.MAX_VALUE;
        }
        return computeDistance(probe, average);
    }

    private double computeDistance(FeatureVector v1, FeatureVector v2) {
        switch (distanceMetric) {
            case EUCLIDEAN:
                return v1.euclideanDistance(v2);
            case COSINE:
                return v1.cosineDistance(v2);
            case MANHATTAN:
                return v1.manhattanDistance(v2);
            case CHI_SQUARE:
                return v1.chiSquareDistance(v2);
            default:
                return v1.euclideanDistance(v2);
        }
    }

    /**
     * Maps a raw distance to a confidence in {@code [0, 1]} via exponential decay:
     * {@code confidence = exp(-distance / scale)}.
     *
     * <p>The per-metric scale constants returned by {@link #getDistanceScale()} are
     * <em>calibration defaults</em> chosen so that typical raw distances on 48x64
     * Eigenfaces features land around the 0.6 recognition threshold. They are not
     * universal — if you swap extractor, image size, or feature dimension, you
     * will want to recalibrate them for your dataset. The classical way is to
     * compute the distribution of same-identity vs different-identity distances
     * on a validation set and choose {@code scale} such that {@code exp(-μ/scale)}
     * lands on the desired operating point.</p>
     *
     * <p>This is deliberately a single line of code so the calibration stays
     * obvious: don't replace it with an opaque callback unless you have a
     * reason to.</p>
     *
     * @param distance the raw distance produced by {@link #computeDistance}
     * @return a confidence score in {@code [0, 1]}
     */
    private double distanceToConfidence(double distance) {
        return Math.exp(-distance / getDistanceScale());
    }

    /**
     * Scale constants used by {@link #distanceToConfidence(double)}. These
     * numbers come from empirical measurement on classical extractors —
     * Eigenfaces/Fisherfaces raw Euclidean distances between aligned 48x64
     * grayscale faces cluster in the 1000–10000 range, so 5000 is a decent
     * half-decay. LBPH chi-square and cosine distances live on very different
     * scales.
     */
    private double getDistanceScale() {
        switch (distanceMetric) {
            case EUCLIDEAN:
                return 5000.0;
            case COSINE:
                return 0.5;
            case MANHATTAN:
                return 10000.0;
            case CHI_SQUARE:
                return 1.0;
            default:
                return 5000.0;
        }
    }

    private List<RecognitionResult.MatchResult> buildAlternatives(List<DistanceEntry> entries, int maxCount) {
        List<RecognitionResult.MatchResult> results = new ArrayList<>();
        Set<String> seen = new HashSet<>();

        for (DistanceEntry entry : entries) {
            if (seen.contains(entry.identity.getId())) continue;
            seen.add(entry.identity.getId());

            results.add(new RecognitionResult.MatchResult(
                entry.identity,
                distanceToConfidence(entry.distance),
                entry.distance
            ));

            if (results.size() >= maxCount) break;
        }

        return results;
    }

    private static class DistanceEntry {
        final Identity identity;
        final double distance;

        DistanceEntry(Identity identity, double distance) {
            this.identity = identity;
            this.distance = distance;
        }
    }

    @Override
    public String toString() {
        return String.format("KNNClassifier{k=%d, metric=%s, enrolled=%d}",
            config.getK(), distanceMetric, enrolledIdentities.size());
    }
}
