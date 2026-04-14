package com.facerecognition.domain.model;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.*;
import java.util.stream.Collectors;

/**
 * Represents the result of a face recognition operation.
 * Contains the matched identity (if any), confidence scores,
 * and alternative matches.
 *
 * <p>Recognition results include:</p>
 * <ul>
 *   <li>The best matching identity (or null if no match)</li>
 *   <li>Confidence score for the match</li>
 *   <li>Alternative candidates ranked by similarity</li>
 *   <li>Processing time and quality metrics</li>
 * </ul>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
public class RecognitionResult implements Serializable {

    private static final long serialVersionUID = 1L;

    private final String requestId;
    private final LocalDateTime timestamp;
    private final MatchResult bestMatch;
    private final List<MatchResult> alternatives;
    private final FaceRegion detectedFace;
    private final FeatureVector extractedFeatures;
    private final ProcessingMetrics metrics;
    private final Status status;
    private final String errorMessage;

    /**
     * Recognition result status.
     */
    public enum Status {
        /** Face recognized successfully. */
        RECOGNIZED,
        /** Face detected but not recognized (unknown person). */
        UNKNOWN,
        /** No face detected in the image. */
        NO_FACE_DETECTED,
        /** Multiple faces detected (ambiguous). */
        MULTIPLE_FACES,
        /** Image quality too low for recognition. */
        POOR_QUALITY,
        /** Recognition failed due to an error. */
        ERROR
    }

    /**
     * Represents a single match candidate.
     */
    public static class MatchResult implements Serializable, Comparable<MatchResult> {
        private static final long serialVersionUID = 1L;

        private final Identity identity;
        private final double confidence;
        private final double distance;

        public MatchResult(Identity identity, double confidence, double distance) {
            this.identity = Objects.requireNonNull(identity);
            this.confidence = confidence;
            this.distance = distance;
        }

        public Identity getIdentity() { return identity; }
        public double getConfidence() { return confidence; }
        public double getDistance() { return distance; }

        @Override
        public int compareTo(MatchResult other) {
            // Higher confidence first
            return Double.compare(other.confidence, this.confidence);
        }

        @Override
        public String toString() {
            return String.format("Match{identity='%s', confidence=%.3f, distance=%.4f}",
                identity.getName(), confidence, distance);
        }
    }

    /**
     * Processing time metrics.
     */
    public static class ProcessingMetrics implements Serializable {
        private static final long serialVersionUID = 1L;

        private final long detectionTimeMs;
        private final long extractionTimeMs;
        private final long matchingTimeMs;
        private final long totalTimeMs;

        public ProcessingMetrics(long detectionTimeMs, long extractionTimeMs,
                                long matchingTimeMs, long totalTimeMs) {
            this.detectionTimeMs = detectionTimeMs;
            this.extractionTimeMs = extractionTimeMs;
            this.matchingTimeMs = matchingTimeMs;
            this.totalTimeMs = totalTimeMs;
        }

        public long getDetectionTimeMs() { return detectionTimeMs; }
        public long getExtractionTimeMs() { return extractionTimeMs; }
        public long getMatchingTimeMs() { return matchingTimeMs; }
        public long getTotalTimeMs() { return totalTimeMs; }

        @Override
        public String toString() {
            return String.format("Metrics{total=%dms, detect=%dms, extract=%dms, match=%dms}",
                totalTimeMs, detectionTimeMs, extractionTimeMs, matchingTimeMs);
        }
    }

    /**
     * Builder for creating RecognitionResult instances.
     */
    public static class Builder {
        private MatchResult bestMatch;
        private List<MatchResult> alternatives = new ArrayList<>();
        private FaceRegion detectedFace;
        private FeatureVector extractedFeatures;
        private ProcessingMetrics metrics;
        private Status status = Status.UNKNOWN;
        private String errorMessage;

        public Builder status(Status status) {
            this.status = status;
            return this;
        }

        public Builder bestMatch(MatchResult match) {
            this.bestMatch = match;
            return this;
        }

        public Builder addAlternative(MatchResult match) {
            this.alternatives.add(match);
            return this;
        }

        public Builder alternatives(List<MatchResult> alternatives) {
            this.alternatives = new ArrayList<>(alternatives);
            return this;
        }

        public Builder detectedFace(FaceRegion face) {
            this.detectedFace = face;
            return this;
        }

        public Builder extractedFeatures(FeatureVector features) {
            this.extractedFeatures = features;
            return this;
        }

        public Builder metrics(ProcessingMetrics metrics) {
            this.metrics = metrics;
            return this;
        }

        public Builder error(String message) {
            this.status = Status.ERROR;
            this.errorMessage = message;
            return this;
        }

        public RecognitionResult build() {
            return new RecognitionResult(this);
        }
    }

    private RecognitionResult(Builder builder) {
        this.requestId = UUID.randomUUID().toString();
        this.timestamp = LocalDateTime.now();
        this.bestMatch = builder.bestMatch;
        this.alternatives = Collections.unmodifiableList(new ArrayList<>(builder.alternatives));
        this.detectedFace = builder.detectedFace;
        this.extractedFeatures = builder.extractedFeatures;
        this.metrics = builder.metrics;
        this.status = builder.status;
        this.errorMessage = builder.errorMessage;
    }

    /**
     * Creates a builder for constructing RecognitionResult.
     *
     * @return a new Builder
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Creates a "recognized" result.
     *
     * @param identity the matched identity
     * @param confidence the confidence score
     * @param distance the distance to the matched sample
     * @return a new RecognitionResult
     */
    public static RecognitionResult recognized(Identity identity, double confidence, double distance) {
        return builder()
            .status(Status.RECOGNIZED)
            .bestMatch(new MatchResult(identity, confidence, distance))
            .build();
    }

    /**
     * Creates an "unknown" result (face detected but not recognized).
     *
     * @return a new RecognitionResult
     */
    public static RecognitionResult unknown() {
        return builder()
            .status(Status.UNKNOWN)
            .build();
    }

    /**
     * Creates a "no face detected" result.
     *
     * @return a new RecognitionResult
     */
    public static RecognitionResult noFaceDetected() {
        return builder()
            .status(Status.NO_FACE_DETECTED)
            .build();
    }

    /**
     * Creates an "error" result.
     *
     * @param message the error message
     * @return a new RecognitionResult
     */
    public static RecognitionResult error(String message) {
        return builder()
            .error(message)
            .build();
    }

    public String getRequestId() {
        return requestId;
    }

    public LocalDateTime getTimestamp() {
        return timestamp;
    }

    public Status getStatus() {
        return status;
    }

    /**
     * Checks if a face was recognized.
     *
     * @return true if status is RECOGNIZED
     */
    public boolean isRecognized() {
        return status == Status.RECOGNIZED;
    }

    /**
     * Checks if a face was detected (regardless of recognition).
     *
     * @return true if a face was detected
     */
    public boolean isFaceDetected() {
        return status == Status.RECOGNIZED || status == Status.UNKNOWN;
    }

    /**
     * Checks if the result represents an error.
     *
     * @return true if status is ERROR
     */
    public boolean isError() {
        return status == Status.ERROR;
    }

    /**
     * Gets the best match result.
     *
     * @return Optional containing the best match, or empty if no match
     */
    public Optional<MatchResult> getBestMatch() {
        return Optional.ofNullable(bestMatch);
    }

    /**
     * Gets the matched identity (convenience method).
     *
     * @return Optional containing the matched identity, or empty
     */
    public Optional<Identity> getIdentity() {
        return getBestMatch().map(MatchResult::getIdentity);
    }

    /**
     * Gets the confidence score of the best match.
     *
     * @return the confidence score, or 0 if no match
     */
    public double getConfidence() {
        return bestMatch != null ? bestMatch.getConfidence() : 0.0;
    }

    /**
     * Gets the distance to the best match.
     *
     * @return the distance, or {@link Double#MAX_VALUE} if no match. Prefer
     *         {@link #getDistanceOpt()} for new code — the sentinel is kept
     *         only for backwards compatibility with earlier callers.
     */
    public double getDistance() {
        return bestMatch != null ? bestMatch.getDistance() : Double.MAX_VALUE;
    }

    /**
     * Gets the distance to the best match as an {@link Optional}. Empty when
     * there is no best match, avoiding the {@link Double#MAX_VALUE} sentinel
     * returned by {@link #getDistance()} that can silently corrupt ranking
     * and threshold logic.
     *
     * @return Optional distance to the best match
     */
    public Optional<Double> getDistanceOpt() {
        return bestMatch != null ? Optional.of(bestMatch.getDistance()) : Optional.empty();
    }

    /**
     * Gets alternative match candidates.
     *
     * @return unmodifiable list of alternatives
     */
    public List<MatchResult> getAlternatives() {
        return alternatives;
    }

    /**
     * Gets the top N alternatives.
     *
     * @param n the number of alternatives
     * @return list of top N alternatives
     */
    public List<MatchResult> getTopAlternatives(int n) {
        return alternatives.stream()
            .sorted()
            .limit(n)
            .collect(Collectors.toList());
    }

    /**
     * Gets the detected face region.
     *
     * @return Optional containing the face region, or empty
     */
    public Optional<FaceRegion> getDetectedFace() {
        return Optional.ofNullable(detectedFace);
    }

    /**
     * Gets the extracted feature vector.
     *
     * @return Optional containing the features, or empty
     */
    public Optional<FeatureVector> getExtractedFeatures() {
        return Optional.ofNullable(extractedFeatures);
    }

    /**
     * Gets processing metrics.
     *
     * @return Optional containing metrics, or empty
     */
    public Optional<ProcessingMetrics> getMetrics() {
        return Optional.ofNullable(metrics);
    }

    /**
     * Gets the error message if status is ERROR.
     *
     * @return Optional containing the error message, or empty
     */
    public Optional<String> getErrorMessage() {
        return Optional.ofNullable(errorMessage);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append("RecognitionResult{status=").append(status);

        if (isRecognized() && bestMatch != null) {
            sb.append(", identity='").append(bestMatch.getIdentity().getName())
              .append("', confidence=").append(String.format("%.3f", bestMatch.getConfidence()));
        }

        if (metrics != null) {
            sb.append(", time=").append(metrics.getTotalTimeMs()).append("ms");
        }

        sb.append("}");
        return sb.toString();
    }
}
