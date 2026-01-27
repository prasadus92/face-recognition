package com.facerecognition.api.rest.dto;

import io.swagger.v3.oas.annotations.media.Schema;

import javax.validation.constraints.DecimalMax;
import javax.validation.constraints.DecimalMin;
import javax.validation.constraints.Max;
import javax.validation.constraints.Min;

/**
 * Request DTO for face recognition operations.
 *
 * <p>This class encapsulates the parameters for a recognition request.
 * The actual image data is sent as a multipart file separately.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Schema(description = "Request parameters for face recognition")
public class RecognitionRequest {

    @Schema(
        description = "Minimum confidence threshold for recognition (0.0 to 1.0)",
        example = "0.6",
        defaultValue = "0.6"
    )
    @DecimalMin(value = "0.0", message = "Threshold must be at least 0.0")
    @DecimalMax(value = "1.0", message = "Threshold must be at most 1.0")
    private Double threshold;

    @Schema(
        description = "Maximum number of alternative matches to return",
        example = "5",
        defaultValue = "5"
    )
    @Min(value = 0, message = "Max alternatives must be at least 0")
    @Max(value = 20, message = "Max alternatives must be at most 20")
    private Integer maxAlternatives;

    @Schema(
        description = "Whether to include feature vector in response",
        example = "false",
        defaultValue = "false"
    )
    private Boolean includeFeatures;

    @Schema(
        description = "Whether to include face region coordinates in response",
        example = "true",
        defaultValue = "true"
    )
    private Boolean includeFaceRegion;

    /**
     * Default constructor.
     */
    public RecognitionRequest() {
        this.threshold = 0.6;
        this.maxAlternatives = 5;
        this.includeFeatures = false;
        this.includeFaceRegion = true;
    }

    /**
     * Gets the recognition threshold.
     *
     * @return the threshold value
     */
    public Double getThreshold() {
        return threshold;
    }

    /**
     * Sets the recognition threshold.
     *
     * @param threshold the threshold value (0.0 to 1.0)
     */
    public void setThreshold(Double threshold) {
        this.threshold = threshold;
    }

    /**
     * Gets the maximum number of alternatives.
     *
     * @return the max alternatives
     */
    public Integer getMaxAlternatives() {
        return maxAlternatives;
    }

    /**
     * Sets the maximum number of alternatives.
     *
     * @param maxAlternatives the max alternatives
     */
    public void setMaxAlternatives(Integer maxAlternatives) {
        this.maxAlternatives = maxAlternatives;
    }

    /**
     * Gets whether to include features in response.
     *
     * @return true if features should be included
     */
    public Boolean getIncludeFeatures() {
        return includeFeatures;
    }

    /**
     * Sets whether to include features in response.
     *
     * @param includeFeatures true to include features
     */
    public void setIncludeFeatures(Boolean includeFeatures) {
        this.includeFeatures = includeFeatures;
    }

    /**
     * Gets whether to include face region in response.
     *
     * @return true if face region should be included
     */
    public Boolean getIncludeFaceRegion() {
        return includeFaceRegion;
    }

    /**
     * Sets whether to include face region in response.
     *
     * @param includeFaceRegion true to include face region
     */
    public void setIncludeFaceRegion(Boolean includeFaceRegion) {
        this.includeFaceRegion = includeFaceRegion;
    }
}
