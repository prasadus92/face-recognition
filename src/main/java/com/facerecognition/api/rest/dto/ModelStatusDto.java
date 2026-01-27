package com.facerecognition.api.rest.dto;

import io.swagger.v3.oas.annotations.media.Schema;

import java.time.LocalDateTime;

/**
 * Data Transfer Object for model status information.
 *
 * <p>This DTO provides comprehensive information about the current
 * state of the face recognition model.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Schema(description = "Current status of the face recognition model")
public class ModelStatusDto {

    /**
     * Possible model states.
     */
    @Schema(description = "Possible model states")
    public enum ModelState {
        /** Model is not initialized. */
        NOT_INITIALIZED,
        /** Model is being trained. */
        TRAINING,
        /** Model is trained and ready for recognition. */
        READY,
        /** Model needs retraining due to new enrollments. */
        NEEDS_TRAINING,
        /** Model training failed. */
        ERROR
    }

    @Schema(description = "Current model state", example = "READY")
    private ModelState state;

    @Schema(description = "Whether the model is ready for recognition", example = "true")
    private boolean ready;

    @Schema(description = "Algorithm name used for feature extraction", example = "Eigenfaces")
    private String algorithmName;

    @Schema(description = "Algorithm version", example = "2")
    private int algorithmVersion;

    @Schema(description = "Number of enrolled identities", example = "50")
    private int identityCount;

    @Schema(description = "Total number of enrolled face samples", example = "150")
    private int totalSampleCount;

    @Schema(description = "Feature vector dimension", example = "128")
    private int featureDimension;

    @Schema(description = "Timestamp of last training", example = "2024-01-15T10:30:00")
    private LocalDateTime lastTrainedAt;

    @Schema(description = "Duration of last training in milliseconds", example = "5000")
    private Long lastTrainingDurationMs;

    @Schema(description = "Number of pending enrollments since last training", example = "5")
    private int pendingEnrollments;

    @Schema(description = "Recognition threshold currently configured", example = "0.6")
    private double recognitionThreshold;

    @Schema(description = "Error message if model is in ERROR state")
    private String errorMessage;

    @Schema(description = "Classifier type used", example = "KNN")
    private String classifierType;

    @Schema(description = "Model file path if persisted", example = "/data/models/face-model.dat")
    private String modelPath;

    @Schema(description = "Model file size in bytes", example = "1048576")
    private Long modelSizeBytes;

    /**
     * Default constructor.
     */
    public ModelStatusDto() {
        this.state = ModelState.NOT_INITIALIZED;
        this.ready = false;
    }

    /**
     * Creates a "not initialized" status.
     *
     * @return a new ModelStatusDto
     */
    public static ModelStatusDto notInitialized() {
        ModelStatusDto status = new ModelStatusDto();
        status.setState(ModelState.NOT_INITIALIZED);
        status.setReady(false);
        return status;
    }

    /**
     * Creates a "ready" status.
     *
     * @param identityCount number of identities
     * @param sampleCount total samples
     * @return a new ModelStatusDto
     */
    public static ModelStatusDto ready(int identityCount, int sampleCount) {
        ModelStatusDto status = new ModelStatusDto();
        status.setState(ModelState.READY);
        status.setReady(true);
        status.setIdentityCount(identityCount);
        status.setTotalSampleCount(sampleCount);
        return status;
    }

    /**
     * Creates a "needs training" status.
     *
     * @param pendingEnrollments number of pending enrollments
     * @return a new ModelStatusDto
     */
    public static ModelStatusDto needsTraining(int pendingEnrollments) {
        ModelStatusDto status = new ModelStatusDto();
        status.setState(ModelState.NEEDS_TRAINING);
        status.setReady(false);
        status.setPendingEnrollments(pendingEnrollments);
        return status;
    }

    /**
     * Creates an "error" status.
     *
     * @param errorMessage the error message
     * @return a new ModelStatusDto
     */
    public static ModelStatusDto error(String errorMessage) {
        ModelStatusDto status = new ModelStatusDto();
        status.setState(ModelState.ERROR);
        status.setReady(false);
        status.setErrorMessage(errorMessage);
        return status;
    }

    // Getters and Setters

    public ModelState getState() { return state; }
    public void setState(ModelState state) { this.state = state; }

    public boolean isReady() { return ready; }
    public void setReady(boolean ready) { this.ready = ready; }

    public String getAlgorithmName() { return algorithmName; }
    public void setAlgorithmName(String algorithmName) { this.algorithmName = algorithmName; }

    public int getAlgorithmVersion() { return algorithmVersion; }
    public void setAlgorithmVersion(int algorithmVersion) { this.algorithmVersion = algorithmVersion; }

    public int getIdentityCount() { return identityCount; }
    public void setIdentityCount(int identityCount) { this.identityCount = identityCount; }

    public int getTotalSampleCount() { return totalSampleCount; }
    public void setTotalSampleCount(int totalSampleCount) { this.totalSampleCount = totalSampleCount; }

    public int getFeatureDimension() { return featureDimension; }
    public void setFeatureDimension(int featureDimension) { this.featureDimension = featureDimension; }

    public LocalDateTime getLastTrainedAt() { return lastTrainedAt; }
    public void setLastTrainedAt(LocalDateTime lastTrainedAt) { this.lastTrainedAt = lastTrainedAt; }

    public Long getLastTrainingDurationMs() { return lastTrainingDurationMs; }
    public void setLastTrainingDurationMs(Long lastTrainingDurationMs) { this.lastTrainingDurationMs = lastTrainingDurationMs; }

    public int getPendingEnrollments() { return pendingEnrollments; }
    public void setPendingEnrollments(int pendingEnrollments) { this.pendingEnrollments = pendingEnrollments; }

    public double getRecognitionThreshold() { return recognitionThreshold; }
    public void setRecognitionThreshold(double recognitionThreshold) { this.recognitionThreshold = recognitionThreshold; }

    public String getErrorMessage() { return errorMessage; }
    public void setErrorMessage(String errorMessage) { this.errorMessage = errorMessage; }

    public String getClassifierType() { return classifierType; }
    public void setClassifierType(String classifierType) { this.classifierType = classifierType; }

    public String getModelPath() { return modelPath; }
    public void setModelPath(String modelPath) { this.modelPath = modelPath; }

    public Long getModelSizeBytes() { return modelSizeBytes; }
    public void setModelSizeBytes(Long modelSizeBytes) { this.modelSizeBytes = modelSizeBytes; }
}
