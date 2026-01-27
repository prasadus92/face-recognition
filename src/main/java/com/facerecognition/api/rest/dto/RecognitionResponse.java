package com.facerecognition.api.rest.dto;

import com.facerecognition.domain.model.RecognitionResult;
import io.swagger.v3.oas.annotations.media.Schema;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Response DTO for face recognition operations.
 *
 * <p>Contains the recognition result including the matched identity,
 * confidence score, alternative matches, and processing metrics.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Schema(description = "Response containing face recognition results")
public class RecognitionResponse {

    @Schema(description = "Unique request identifier", example = "550e8400-e29b-41d4-a716-446655440000")
    private String requestId;

    @Schema(description = "Timestamp of the recognition", example = "2024-01-15T10:30:00")
    private LocalDateTime timestamp;

    @Schema(description = "Recognition status", example = "RECOGNIZED")
    private String status;

    @Schema(description = "Whether a face was successfully recognized", example = "true")
    private boolean recognized;

    @Schema(description = "Best match identity details")
    private MatchDto bestMatch;

    @Schema(description = "Alternative match candidates")
    private List<MatchDto> alternatives;

    @Schema(description = "Detected face region coordinates")
    private FaceRegionDto faceRegion;

    @Schema(description = "Processing time metrics")
    private ProcessingMetricsDto metrics;

    @Schema(description = "Extracted feature vector (if requested)")
    private double[] features;

    /**
     * DTO for a single match result.
     */
    @Schema(description = "A single identity match result")
    public static class MatchDto {

        @Schema(description = "Identity ID", example = "550e8400-e29b-41d4-a716-446655440000")
        private String identityId;

        @Schema(description = "Identity name", example = "John Doe")
        private String name;

        @Schema(description = "Confidence score (0.0 to 1.0)", example = "0.95")
        private double confidence;

        @Schema(description = "Distance to matched sample", example = "0.123")
        private double distance;

        public MatchDto() {}

        public MatchDto(String identityId, String name, double confidence, double distance) {
            this.identityId = identityId;
            this.name = name;
            this.confidence = confidence;
            this.distance = distance;
        }

        public String getIdentityId() { return identityId; }
        public void setIdentityId(String identityId) { this.identityId = identityId; }

        public String getName() { return name; }
        public void setName(String name) { this.name = name; }

        public double getConfidence() { return confidence; }
        public void setConfidence(double confidence) { this.confidence = confidence; }

        public double getDistance() { return distance; }
        public void setDistance(double distance) { this.distance = distance; }
    }

    /**
     * DTO for face region coordinates.
     */
    @Schema(description = "Detected face region coordinates")
    public static class FaceRegionDto {

        @Schema(description = "X coordinate of top-left corner", example = "100")
        private int x;

        @Schema(description = "Y coordinate of top-left corner", example = "50")
        private int y;

        @Schema(description = "Width of face region", example = "200")
        private int width;

        @Schema(description = "Height of face region", example = "250")
        private int height;

        public FaceRegionDto() {}

        public FaceRegionDto(int x, int y, int width, int height) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
        }

        public int getX() { return x; }
        public void setX(int x) { this.x = x; }

        public int getY() { return y; }
        public void setY(int y) { this.y = y; }

        public int getWidth() { return width; }
        public void setWidth(int width) { this.width = width; }

        public int getHeight() { return height; }
        public void setHeight(int height) { this.height = height; }
    }

    /**
     * DTO for processing time metrics.
     */
    @Schema(description = "Processing time metrics in milliseconds")
    public static class ProcessingMetricsDto {

        @Schema(description = "Face detection time (ms)", example = "50")
        private long detectionTimeMs;

        @Schema(description = "Feature extraction time (ms)", example = "100")
        private long extractionTimeMs;

        @Schema(description = "Matching/classification time (ms)", example = "20")
        private long matchingTimeMs;

        @Schema(description = "Total processing time (ms)", example = "170")
        private long totalTimeMs;

        public ProcessingMetricsDto() {}

        public ProcessingMetricsDto(long detectionTimeMs, long extractionTimeMs,
                                   long matchingTimeMs, long totalTimeMs) {
            this.detectionTimeMs = detectionTimeMs;
            this.extractionTimeMs = extractionTimeMs;
            this.matchingTimeMs = matchingTimeMs;
            this.totalTimeMs = totalTimeMs;
        }

        public long getDetectionTimeMs() { return detectionTimeMs; }
        public void setDetectionTimeMs(long detectionTimeMs) { this.detectionTimeMs = detectionTimeMs; }

        public long getExtractionTimeMs() { return extractionTimeMs; }
        public void setExtractionTimeMs(long extractionTimeMs) { this.extractionTimeMs = extractionTimeMs; }

        public long getMatchingTimeMs() { return matchingTimeMs; }
        public void setMatchingTimeMs(long matchingTimeMs) { this.matchingTimeMs = matchingTimeMs; }

        public long getTotalTimeMs() { return totalTimeMs; }
        public void setTotalTimeMs(long totalTimeMs) { this.totalTimeMs = totalTimeMs; }
    }

    /**
     * Default constructor.
     */
    public RecognitionResponse() {
        this.alternatives = new ArrayList<>();
    }

    /**
     * Creates a RecognitionResponse from a domain RecognitionResult.
     *
     * @param result the domain recognition result
     * @return a new RecognitionResponse
     */
    public static RecognitionResponse fromDomain(RecognitionResult result) {
        RecognitionResponse response = new RecognitionResponse();
        response.setRequestId(result.getRequestId());
        response.setTimestamp(result.getTimestamp());
        response.setStatus(result.getStatus().name());
        response.setRecognized(result.isRecognized());

        result.getBestMatch().ifPresent(match -> {
            response.setBestMatch(new MatchDto(
                match.getIdentity().getId(),
                match.getIdentity().getName(),
                match.getConfidence(),
                match.getDistance()
            ));
        });

        List<MatchDto> alternatives = new ArrayList<>();
        for (RecognitionResult.MatchResult alt : result.getAlternatives()) {
            alternatives.add(new MatchDto(
                alt.getIdentity().getId(),
                alt.getIdentity().getName(),
                alt.getConfidence(),
                alt.getDistance()
            ));
        }
        response.setAlternatives(alternatives);

        result.getDetectedFace().ifPresent(face -> {
            response.setFaceRegion(new FaceRegionDto(
                face.getX(), face.getY(),
                face.getWidth(), face.getHeight()
            ));
        });

        result.getMetrics().ifPresent(metrics -> {
            response.setMetrics(new ProcessingMetricsDto(
                metrics.getDetectionTimeMs(),
                metrics.getExtractionTimeMs(),
                metrics.getMatchingTimeMs(),
                metrics.getTotalTimeMs()
            ));
        });

        result.getExtractedFeatures().ifPresent(features -> {
            response.setFeatures(features.getFeatures());
        });

        return response;
    }

    // Getters and Setters

    public String getRequestId() { return requestId; }
    public void setRequestId(String requestId) { this.requestId = requestId; }

    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }

    public String getStatus() { return status; }
    public void setStatus(String status) { this.status = status; }

    public boolean isRecognized() { return recognized; }
    public void setRecognized(boolean recognized) { this.recognized = recognized; }

    public MatchDto getBestMatch() { return bestMatch; }
    public void setBestMatch(MatchDto bestMatch) { this.bestMatch = bestMatch; }

    public List<MatchDto> getAlternatives() { return alternatives; }
    public void setAlternatives(List<MatchDto> alternatives) { this.alternatives = alternatives; }

    public FaceRegionDto getFaceRegion() { return faceRegion; }
    public void setFaceRegion(FaceRegionDto faceRegion) { this.faceRegion = faceRegion; }

    public ProcessingMetricsDto getMetrics() { return metrics; }
    public void setMetrics(ProcessingMetricsDto metrics) { this.metrics = metrics; }

    public double[] getFeatures() { return features; }
    public void setFeatures(double[] features) { this.features = features; }
}
