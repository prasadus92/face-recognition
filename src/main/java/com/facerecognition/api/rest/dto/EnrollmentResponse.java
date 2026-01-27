package com.facerecognition.api.rest.dto;

import com.facerecognition.domain.model.Identity;
import io.swagger.v3.oas.annotations.media.Schema;

import java.time.LocalDateTime;

/**
 * Response DTO for face enrollment operations.
 *
 * <p>Contains the result of an enrollment operation including
 * the created/updated identity and quality metrics.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Schema(description = "Response containing face enrollment results")
public class EnrollmentResponse {

    @Schema(description = "Whether the enrollment was successful", example = "true")
    private boolean success;

    @Schema(description = "Status message", example = "Face enrolled successfully")
    private String message;

    @Schema(description = "Enrolled identity ID", example = "550e8400-e29b-41d4-a716-446655440000")
    private String identityId;

    @Schema(description = "Identity name", example = "John Doe")
    private String name;

    @Schema(description = "External system ID", example = "EMP-12345")
    private String externalId;

    @Schema(description = "Sample ID of the enrolled face", example = "660e8400-e29b-41d4-a716-446655440001")
    private String sampleId;

    @Schema(description = "Total number of enrolled samples for this identity", example = "3")
    private int sampleCount;

    @Schema(description = "Quality score of the enrolled face (0.0 to 1.0)", example = "0.85")
    private double qualityScore;

    @Schema(description = "Timestamp of enrollment", example = "2024-01-15T10:30:00")
    private LocalDateTime enrolledAt;

    @Schema(description = "Whether the model needs retraining after this enrollment", example = "true")
    private boolean requiresTraining;

    /**
     * Default constructor.
     */
    public EnrollmentResponse() {
    }

    /**
     * Creates a successful enrollment response.
     *
     * @param identity the enrolled identity
     * @param sampleId the ID of the enrolled sample
     * @param qualityScore the quality score
     * @return a new EnrollmentResponse
     */
    public static EnrollmentResponse success(Identity identity, String sampleId, double qualityScore) {
        EnrollmentResponse response = new EnrollmentResponse();
        response.setSuccess(true);
        response.setMessage("Face enrolled successfully");
        response.setIdentityId(identity.getId());
        response.setName(identity.getName());
        response.setExternalId(identity.getExternalId());
        response.setSampleId(sampleId);
        response.setSampleCount(identity.getSampleCount());
        response.setQualityScore(qualityScore);
        response.setEnrolledAt(LocalDateTime.now());
        response.setRequiresTraining(true);
        return response;
    }

    /**
     * Creates a failed enrollment response.
     *
     * @param message the failure message
     * @return a new EnrollmentResponse
     */
    public static EnrollmentResponse failure(String message) {
        EnrollmentResponse response = new EnrollmentResponse();
        response.setSuccess(false);
        response.setMessage(message);
        return response;
    }

    // Getters and Setters

    public boolean isSuccess() {
        return success;
    }

    public void setSuccess(boolean success) {
        this.success = success;
    }

    public String getMessage() {
        return message;
    }

    public void setMessage(String message) {
        this.message = message;
    }

    public String getIdentityId() {
        return identityId;
    }

    public void setIdentityId(String identityId) {
        this.identityId = identityId;
    }

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getExternalId() {
        return externalId;
    }

    public void setExternalId(String externalId) {
        this.externalId = externalId;
    }

    public String getSampleId() {
        return sampleId;
    }

    public void setSampleId(String sampleId) {
        this.sampleId = sampleId;
    }

    public int getSampleCount() {
        return sampleCount;
    }

    public void setSampleCount(int sampleCount) {
        this.sampleCount = sampleCount;
    }

    public double getQualityScore() {
        return qualityScore;
    }

    public void setQualityScore(double qualityScore) {
        this.qualityScore = qualityScore;
    }

    public LocalDateTime getEnrolledAt() {
        return enrolledAt;
    }

    public void setEnrolledAt(LocalDateTime enrolledAt) {
        this.enrolledAt = enrolledAt;
    }

    public boolean isRequiresTraining() {
        return requiresTraining;
    }

    public void setRequiresTraining(boolean requiresTraining) {
        this.requiresTraining = requiresTraining;
    }
}
