package com.facerecognition.api.rest.dto;

import com.facerecognition.domain.model.Identity;
import io.swagger.v3.oas.annotations.media.Schema;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Data Transfer Object for Identity information.
 *
 * <p>This DTO represents the full details of an enrolled identity
 * including all samples and metadata.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Schema(description = "Detailed information about an enrolled identity")
public class IdentityDto {

    @Schema(description = "Unique identity identifier", example = "550e8400-e29b-41d4-a716-446655440000")
    private String id;

    @Schema(description = "Identity name", example = "John Doe")
    private String name;

    @Schema(description = "External system identifier", example = "EMP-12345")
    private String externalId;

    @Schema(description = "Whether the identity is active", example = "true")
    private boolean active;

    @Schema(description = "Creation timestamp", example = "2024-01-15T10:30:00")
    private LocalDateTime createdAt;

    @Schema(description = "Last update timestamp", example = "2024-01-15T14:45:00")
    private LocalDateTime updatedAt;

    @Schema(description = "Number of enrolled face samples", example = "3")
    private int sampleCount;

    @Schema(description = "Average quality score across all samples", example = "0.85")
    private double averageQualityScore;

    @Schema(description = "List of enrolled face samples")
    private List<SampleDto> samples;

    @Schema(description = "Additional metadata as key-value pairs")
    private Map<String, String> metadata;

    /**
     * DTO for an enrolled face sample.
     */
    @Schema(description = "Information about an enrolled face sample")
    public static class SampleDto {

        @Schema(description = "Sample identifier", example = "660e8400-e29b-41d4-a716-446655440001")
        private String sampleId;

        @Schema(description = "Enrollment timestamp", example = "2024-01-15T10:30:00")
        private LocalDateTime enrolledAt;

        @Schema(description = "Quality score of the sample", example = "0.87")
        private double qualityScore;

        @Schema(description = "Description of the source image", example = "Employee badge photo")
        private String sourceDescription;

        @Schema(description = "Feature vector dimension", example = "128")
        private int featureDimension;

        public SampleDto() {}

        public String getSampleId() { return sampleId; }
        public void setSampleId(String sampleId) { this.sampleId = sampleId; }

        public LocalDateTime getEnrolledAt() { return enrolledAt; }
        public void setEnrolledAt(LocalDateTime enrolledAt) { this.enrolledAt = enrolledAt; }

        public double getQualityScore() { return qualityScore; }
        public void setQualityScore(double qualityScore) { this.qualityScore = qualityScore; }

        public String getSourceDescription() { return sourceDescription; }
        public void setSourceDescription(String sourceDescription) { this.sourceDescription = sourceDescription; }

        public int getFeatureDimension() { return featureDimension; }
        public void setFeatureDimension(int featureDimension) { this.featureDimension = featureDimension; }
    }

    /**
     * Default constructor.
     */
    public IdentityDto() {
        this.samples = new ArrayList<>();
        this.metadata = new HashMap<>();
    }

    /**
     * Creates an IdentityDto from a domain Identity.
     *
     * @param identity the domain identity
     * @return a new IdentityDto
     */
    public static IdentityDto fromDomain(Identity identity) {
        return fromDomain(identity, true);
    }

    /**
     * Creates an IdentityDto from a domain Identity.
     *
     * @param identity the domain identity
     * @param includeSamples whether to include sample details
     * @return a new IdentityDto
     */
    public static IdentityDto fromDomain(Identity identity, boolean includeSamples) {
        IdentityDto dto = new IdentityDto();
        dto.setId(identity.getId());
        dto.setName(identity.getName());
        dto.setExternalId(identity.getExternalId());
        dto.setActive(identity.isActive());
        dto.setCreatedAt(identity.getCreatedAt());
        dto.setUpdatedAt(identity.getUpdatedAt());
        dto.setSampleCount(identity.getSampleCount());
        dto.setAverageQualityScore(identity.getAverageQualityScore());
        dto.setMetadata(new HashMap<>(identity.getAllMetadata()));

        if (includeSamples) {
            List<SampleDto> samples = new ArrayList<>();
            for (Identity.EnrolledSample sample : identity.getSamples()) {
                SampleDto sampleDto = new SampleDto();
                sampleDto.setSampleId(sample.getSampleId());
                sampleDto.setEnrolledAt(sample.getEnrolledAt());
                sampleDto.setQualityScore(sample.getQualityScore());
                sampleDto.setSourceDescription(sample.getSourceDescription());
                sampleDto.setFeatureDimension(sample.getFeatures().getDimension());
                samples.add(sampleDto);
            }
            dto.setSamples(samples);
        }

        return dto;
    }

    /**
     * Creates a summary IdentityDto (without sample details).
     *
     * @param identity the domain identity
     * @return a new IdentityDto summary
     */
    public static IdentityDto summary(Identity identity) {
        return fromDomain(identity, false);
    }

    // Getters and Setters

    public String getId() { return id; }
    public void setId(String id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getExternalId() { return externalId; }
    public void setExternalId(String externalId) { this.externalId = externalId; }

    public boolean isActive() { return active; }
    public void setActive(boolean active) { this.active = active; }

    public LocalDateTime getCreatedAt() { return createdAt; }
    public void setCreatedAt(LocalDateTime createdAt) { this.createdAt = createdAt; }

    public LocalDateTime getUpdatedAt() { return updatedAt; }
    public void setUpdatedAt(LocalDateTime updatedAt) { this.updatedAt = updatedAt; }

    public int getSampleCount() { return sampleCount; }
    public void setSampleCount(int sampleCount) { this.sampleCount = sampleCount; }

    public double getAverageQualityScore() { return averageQualityScore; }
    public void setAverageQualityScore(double averageQualityScore) { this.averageQualityScore = averageQualityScore; }

    public List<SampleDto> getSamples() { return samples; }
    public void setSamples(List<SampleDto> samples) { this.samples = samples; }

    public Map<String, String> getMetadata() { return metadata; }
    public void setMetadata(Map<String, String> metadata) { this.metadata = metadata; }
}
