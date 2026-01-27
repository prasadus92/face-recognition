package com.facerecognition.api.rest.dto;

import io.swagger.v3.oas.annotations.media.Schema;

import javax.validation.constraints.NotBlank;
import javax.validation.constraints.Size;
import java.util.HashMap;
import java.util.Map;

/**
 * Request DTO for face enrollment operations.
 *
 * <p>This class encapsulates the parameters for enrolling a new face
 * in the recognition system. The actual image data is sent as a
 * multipart file separately.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Schema(description = "Request parameters for face enrollment")
public class EnrollmentRequest {

    @Schema(
        description = "Name of the person being enrolled",
        example = "John Doe",
        required = true
    )
    @NotBlank(message = "Name is required")
    @Size(min = 1, max = 255, message = "Name must be between 1 and 255 characters")
    private String name;

    @Schema(
        description = "Optional external system identifier",
        example = "EMP-12345"
    )
    @Size(max = 255, message = "External ID must be at most 255 characters")
    private String externalId;

    @Schema(
        description = "Optional description of the image source",
        example = "Employee badge photo"
    )
    @Size(max = 1000, message = "Source description must be at most 1000 characters")
    private String sourceDescription;

    @Schema(
        description = "Additional metadata as key-value pairs",
        example = "{\"department\": \"Engineering\", \"location\": \"Building A\"}"
    )
    private Map<String, String> metadata;

    @Schema(
        description = "Whether to validate image quality before enrollment",
        example = "true",
        defaultValue = "true"
    )
    private Boolean validateQuality;

    @Schema(
        description = "Minimum quality score required for enrollment (0.0 to 1.0)",
        example = "0.5",
        defaultValue = "0.3"
    )
    private Double minQualityScore;

    /**
     * Default constructor.
     */
    public EnrollmentRequest() {
        this.metadata = new HashMap<>();
        this.validateQuality = true;
        this.minQualityScore = 0.3;
    }

    /**
     * Constructor with name.
     *
     * @param name the identity name
     */
    public EnrollmentRequest(String name) {
        this();
        this.name = name;
    }

    /**
     * Gets the identity name.
     *
     * @return the name
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the identity name.
     *
     * @param name the name
     */
    public void setName(String name) {
        this.name = name;
    }

    /**
     * Gets the external ID.
     *
     * @return the external ID
     */
    public String getExternalId() {
        return externalId;
    }

    /**
     * Sets the external ID.
     *
     * @param externalId the external ID
     */
    public void setExternalId(String externalId) {
        this.externalId = externalId;
    }

    /**
     * Gets the source description.
     *
     * @return the source description
     */
    public String getSourceDescription() {
        return sourceDescription;
    }

    /**
     * Sets the source description.
     *
     * @param sourceDescription the source description
     */
    public void setSourceDescription(String sourceDescription) {
        this.sourceDescription = sourceDescription;
    }

    /**
     * Gets the metadata map.
     *
     * @return the metadata
     */
    public Map<String, String> getMetadata() {
        return metadata;
    }

    /**
     * Sets the metadata map.
     *
     * @param metadata the metadata
     */
    public void setMetadata(Map<String, String> metadata) {
        this.metadata = metadata != null ? metadata : new HashMap<>();
    }

    /**
     * Gets whether quality validation is enabled.
     *
     * @return true if quality validation is enabled
     */
    public Boolean getValidateQuality() {
        return validateQuality;
    }

    /**
     * Sets whether quality validation is enabled.
     *
     * @param validateQuality true to enable quality validation
     */
    public void setValidateQuality(Boolean validateQuality) {
        this.validateQuality = validateQuality;
    }

    /**
     * Gets the minimum quality score.
     *
     * @return the minimum quality score
     */
    public Double getMinQualityScore() {
        return minQualityScore;
    }

    /**
     * Sets the minimum quality score.
     *
     * @param minQualityScore the minimum quality score
     */
    public void setMinQualityScore(Double minQualityScore) {
        this.minQualityScore = minQualityScore;
    }
}
