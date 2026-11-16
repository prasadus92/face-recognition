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
    private transient volatile Calibration calibration;

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
        calibration = null;
    }

    @Override
    public boolean unenroll(String identityId) {
        boolean removed = enrolledIdentities.remove(identityId) != null;
        calibration = null;
        return removed;
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
        calibration = null;
    }

    @Override
    public void clear() {
        enrolledIdentities.clear();
        calibration = null;
    }

    @Override
    public void retrain() {
        calibration = Calibration.of(this);
    }

    private List<DistanceEntry> computeAllDistances(FeatureVector probe) {
        List<DistanceEntry> entries = new ArrayList<>();

        for (Identity identity : enrolledIdentities.values()) {
            if (!identity.isActive()) continue;

            entries.add(new DistanceEntry(identity, distanceTo(probe, identity)));
        }

        return entries;
    }

    private double distanceTo(FeatureVector probe, Identity identity) {
        return config.isUseAverageFeatures()
                ? computeDistanceToAverage(probe, identity)
                : computeMinDistance(probe, identity);
    }

    /**
     * Distances measured on the gallery itself: for sampled anchor photos, the distance to their own
     * person ({@code genuine}: the nearest other photo, or the person's average in average mode) and
     * to the nearest other person ({@code impostor}). Confidence is a logistic curve centered halfway
     * between their medians. At most {@value #MAX_ANCHORS} anchors, picked at a fixed stride, are
     * compared with the whole gallery, so the cost grows linearly with the gallery.
     */
    private record Calibration(double genuine, double impostor) {

        static final int MAX_ANCHORS = 500;

        private record Entry(Identity owner, FeatureVector features) {
        }

        static Calibration of(KNNClassifier classifier) {
            List<Identity> active = classifier.enrolledIdentities.values().stream()
                    .filter(Identity::isActive).filter(Identity::hasSamples).toList();
            List<Entry> gallery = new ArrayList<>();
            active.forEach(owner -> owner.getSamples().forEach(s -> gallery.add(new Entry(owner, s.getFeatures()))));
            Map<Identity, FeatureVector> averages = new HashMap<>();
            if (classifier.config.isUseAverageFeatures()) {
                active.forEach(owner -> averages.put(owner, owner.getAverageFeatureVector()));
            }

            int stride = Math.max(1, (gallery.size() + MAX_ANCHORS - 1) / MAX_ANCHORS);
            List<Double> genuine = new ArrayList<>();
            List<Double> impostor = new ArrayList<>();
            for (int a = 0; a < gallery.size(); a += stride) {
                Entry anchor = gallery.get(a);
                double own = Double.MAX_VALUE;
                double other = Double.MAX_VALUE;
                if (averages.isEmpty()) {
                    for (Entry candidate : gallery) {
                        if (candidate == anchor) {
                            continue;
                        }
                        double d = classifier.computeDistance(anchor.features(), candidate.features());
                        if (candidate.owner() == anchor.owner()) {
                            own = Math.min(own, d);
                        } else {
                            other = Math.min(other, d);
                        }
                    }
                } else {
                    for (Map.Entry<Identity, FeatureVector> person : averages.entrySet()) {
                        double d = classifier.computeDistance(anchor.features(), person.getValue());
                        if (person.getKey() == anchor.owner()) {
                            own = anchor.owner().getSampleCount() > 1 ? d : own;
                        } else {
                            other = Math.min(other, d);
                        }
                    }
                }
                if (own < Double.MAX_VALUE) {
                    genuine.add(own);
                }
                if (other < Double.MAX_VALUE) {
                    impostor.add(other);
                }
            }
            return new Calibration(median(genuine), median(impostor));
        }

        /** @return the confidence for a distance, or NaN when the gallery cannot calibrate it */
        double confidence(double distance) {
            if (Double.isNaN(impostor) || impostor <= 0) {
                return Double.NaN;
            }
            // With 1 photo per person there is no genuine distance; 0 keeps strangers rejected.
            double own = Double.isNaN(genuine) || genuine >= impostor ? 0 : genuine;
            double center = (own + impostor) / 2;
            double width = (impostor - own) / 8;
            return 1.0 / (1.0 + Math.exp((distance - center) / width));
        }

        private static double median(List<Double> values) {
            if (values.isEmpty()) {
                return Double.NaN;
            }
            List<Double> sorted = values.stream().sorted().toList();
            int middle = sorted.size() / 2;
            return sorted.size() % 2 == 1 ? sorted.get(middle) : (sorted.get(middle - 1) + sorted.get(middle)) / 2;
        }
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
     * Maps a distance to a confidence in {@code [0, 1]}, calibrated on the gallery (see
     * {@link Calibration}): a probe as close as a typical photo of the same person scores about 0.98,
     * one as far as the typical nearest other person about 0.02, and 0.5 sits halfway. With fewer than
     * 2 people enrolled there is nothing to calibrate against and a fixed per-metric scale applies.
     *
     * @param distance the distance to the best match
     * @return the confidence
     */
    private double distanceToConfidence(double distance) {
        Calibration current = calibration;
        if (current == null) {
            current = Calibration.of(this);
            calibration = current;
        }
        double confidence = current.confidence(distance);
        return Double.isNaN(confidence) ? Math.exp(-distance / fallbackScale()) : confidence;
    }

    private double fallbackScale() {
        return switch (distanceMetric) {
            case COSINE -> 0.5;
            case CHI_SQUARE -> 1.0;
            case MANHATTAN -> 10000.0;
            case EUCLIDEAN, MAHALANOBIS -> 5000.0;
        };
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
