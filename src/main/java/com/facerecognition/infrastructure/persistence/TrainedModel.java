package com.facerecognition.infrastructure.persistence;

import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.model.Identity;

import Jama.Matrix;

import java.io.Serializable;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.*;

/**
 * Serializable container for a trained face recognition model.
 *
 * <p>This class encapsulates all components of a trained model including:</p>
 * <ul>
 *   <li>Algorithm identification and version information</li>
 *   <li>Eigenfaces/eigenvectors matrices for PCA-based methods</li>
 *   <li>Mean face vector for centering operations</li>
 *   <li>Enrolled identities with their feature vectors</li>
 *   <li>Hyperparameters and configuration used during training</li>
 *   <li>Model metadata including training metrics and timestamps</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * TrainedModel model = TrainedModel.builder("Eigenfaces", 2)
 *     .eigenVectors(eigenMatrix)
 *     .meanFace(meanVector)
 *     .addIdentity(identity)
 *     .setHyperparameter("numComponents", 50)
 *     .build();
 *
 * // Save and load
 * repository.save(model, "model.frm");
 * TrainedModel loaded = repository.load("model.frm");
 * }</pre>
 *
 * <h3>Thread Safety:</h3>
 * <p>This class is effectively immutable after construction. The builder
 * pattern ensures all fields are set before the model is used.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see ModelRepository
 * @see ModelSerializer
 */
public class TrainedModel implements Serializable {

    private static final long serialVersionUID = 1L;

    /**
     * Current format version for backward compatibility.
     */
    public static final int FORMAT_VERSION = 1;

    // Algorithm identification
    private final String algorithmName;
    private final int algorithmVersion;
    private final String modelId;

    // Timestamps
    private final long trainingTimestampMillis;
    private final long creationTimestampMillis;

    // Matrix data (stored as 2D arrays for serialization)
    private final double[][] eigenVectorsData;
    private final double[][] eigenValuesData;
    private final double[] meanFaceData;

    // Enrolled identities
    private final List<EnrolledIdentity> enrolledIdentities;

    // Hyperparameters and configuration
    private final Map<String, Object> hyperparameters;
    private final Map<String, Object> configuration;

    // Model metadata
    private final ModelMetadata metadata;

    // Format version for backward compatibility
    private final int formatVersion;

    /**
     * Private constructor. Use {@link Builder} to create instances.
     *
     * @param builder the builder containing all model data
     */
    private TrainedModel(Builder builder) {
        this.algorithmName = builder.algorithmName;
        this.algorithmVersion = builder.algorithmVersion;
        this.modelId = builder.modelId != null ? builder.modelId : UUID.randomUUID().toString();
        this.trainingTimestampMillis = builder.trainingTimestamp != null
                ? builder.trainingTimestamp.atZone(ZoneId.systemDefault()).toInstant().toEpochMilli()
                : System.currentTimeMillis();
        this.creationTimestampMillis = System.currentTimeMillis();

        this.eigenVectorsData = builder.eigenVectorsData;
        this.eigenValuesData = builder.eigenValuesData;
        this.meanFaceData = builder.meanFaceData;

        this.enrolledIdentities = new ArrayList<>(builder.enrolledIdentities);
        this.hyperparameters = new HashMap<>(builder.hyperparameters);
        this.configuration = new HashMap<>(builder.configuration);
        this.metadata = builder.metadata != null ? builder.metadata : new ModelMetadata();
        this.formatVersion = FORMAT_VERSION;
    }

    /**
     * Creates a new builder for constructing a TrainedModel.
     *
     * @param algorithmName the name of the algorithm (e.g., "Eigenfaces", "LBPH")
     * @param algorithmVersion the version of the algorithm
     * @return a new Builder instance
     */
    public static Builder builder(String algorithmName, int algorithmVersion) {
        return new Builder(algorithmName, algorithmVersion);
    }

    /**
     * Gets the algorithm name.
     *
     * @return the algorithm name
     */
    public String getAlgorithmName() {
        return algorithmName;
    }

    /**
     * Gets the algorithm version.
     *
     * @return the algorithm version number
     */
    public int getAlgorithmVersion() {
        return algorithmVersion;
    }

    /**
     * Gets the unique model identifier.
     *
     * @return the model ID
     */
    public String getModelId() {
        return modelId;
    }

    /**
     * Gets the training timestamp.
     *
     * @return the timestamp when training completed
     */
    public LocalDateTime getTrainingTimestamp() {
        return LocalDateTime.ofInstant(
                Instant.ofEpochMilli(trainingTimestampMillis),
                ZoneId.systemDefault());
    }

    /**
     * Gets the model creation timestamp.
     *
     * @return the timestamp when this model object was created
     */
    public LocalDateTime getCreationTimestamp() {
        return LocalDateTime.ofInstant(
                Instant.ofEpochMilli(creationTimestampMillis),
                ZoneId.systemDefault());
    }

    /**
     * Gets the format version used for serialization.
     *
     * @return the format version number
     */
    public int getFormatVersion() {
        return formatVersion;
    }

    /**
     * Checks if this model has eigenface/eigenvector data.
     *
     * @return true if eigenvector matrix is present
     */
    public boolean hasEigenVectors() {
        return eigenVectorsData != null && eigenVectorsData.length > 0;
    }

    /**
     * Gets the eigenvectors matrix.
     *
     * @return a new Matrix containing the eigenvectors, or null if not set
     */
    public Matrix getEigenVectors() {
        if (eigenVectorsData == null) {
            return null;
        }
        return new Matrix(eigenVectorsData);
    }

    /**
     * Gets the eigenvectors as a raw 2D array.
     *
     * @return copy of the eigenvectors data, or null if not set
     */
    public double[][] getEigenVectorsData() {
        if (eigenVectorsData == null) {
            return null;
        }
        return copyArray(eigenVectorsData);
    }

    /**
     * Checks if this model has eigenvalue data.
     *
     * @return true if eigenvalue matrix is present
     */
    public boolean hasEigenValues() {
        return eigenValuesData != null && eigenValuesData.length > 0;
    }

    /**
     * Gets the eigenvalues matrix.
     *
     * @return a new Matrix containing the eigenvalues, or null if not set
     */
    public Matrix getEigenValues() {
        if (eigenValuesData == null) {
            return null;
        }
        return new Matrix(eigenValuesData);
    }

    /**
     * Gets the eigenvalues as a raw 2D array.
     *
     * @return copy of the eigenvalues data, or null if not set
     */
    public double[][] getEigenValuesData() {
        if (eigenValuesData == null) {
            return null;
        }
        return copyArray(eigenValuesData);
    }

    /**
     * Checks if this model has a mean face.
     *
     * @return true if mean face vector is present
     */
    public boolean hasMeanFace() {
        return meanFaceData != null && meanFaceData.length > 0;
    }

    /**
     * Gets the mean face as a column vector Matrix.
     *
     * @return a new Matrix containing the mean face, or null if not set
     */
    public Matrix getMeanFace() {
        if (meanFaceData == null) {
            return null;
        }
        return new Matrix(meanFaceData, meanFaceData.length);
    }

    /**
     * Gets the mean face as a raw array.
     *
     * @return copy of the mean face data, or null if not set
     */
    public double[] getMeanFaceData() {
        if (meanFaceData == null) {
            return null;
        }
        return Arrays.copyOf(meanFaceData, meanFaceData.length);
    }

    /**
     * Gets all enrolled identities.
     *
     * @return unmodifiable list of enrolled identities
     */
    public List<EnrolledIdentity> getEnrolledIdentities() {
        return Collections.unmodifiableList(enrolledIdentities);
    }

    /**
     * Gets the number of enrolled identities.
     *
     * @return the identity count
     */
    public int getIdentityCount() {
        return enrolledIdentities.size();
    }

    /**
     * Finds an enrolled identity by ID.
     *
     * @param identityId the identity ID to search for
     * @return an Optional containing the identity if found
     */
    public Optional<EnrolledIdentity> findIdentity(String identityId) {
        return enrolledIdentities.stream()
                .filter(ei -> ei.getIdentityId().equals(identityId))
                .findFirst();
    }

    /**
     * Gets all hyperparameters.
     *
     * @return unmodifiable map of hyperparameters
     */
    public Map<String, Object> getHyperparameters() {
        return Collections.unmodifiableMap(hyperparameters);
    }

    /**
     * Gets a specific hyperparameter value.
     *
     * @param name the hyperparameter name
     * @param <T> the expected type
     * @return the value, or null if not set
     */
    @SuppressWarnings("unchecked")
    public <T> T getHyperparameter(String name) {
        return (T) hyperparameters.get(name);
    }

    /**
     * Gets a hyperparameter with a default value.
     *
     * @param name the hyperparameter name
     * @param defaultValue the default value if not set
     * @param <T> the expected type
     * @return the value or default
     */
    @SuppressWarnings("unchecked")
    public <T> T getHyperparameter(String name, T defaultValue) {
        Object value = hyperparameters.get(name);
        return value != null ? (T) value : defaultValue;
    }

    /**
     * Gets all configuration settings.
     *
     * @return unmodifiable map of configuration
     */
    public Map<String, Object> getConfiguration() {
        return Collections.unmodifiableMap(configuration);
    }

    /**
     * Gets a specific configuration value.
     *
     * @param name the configuration name
     * @param <T> the expected type
     * @return the value, or null if not set
     */
    @SuppressWarnings("unchecked")
    public <T> T getConfigValue(String name) {
        return (T) configuration.get(name);
    }

    /**
     * Gets the model metadata.
     *
     * @return the metadata object
     */
    public ModelMetadata getMetadata() {
        return metadata;
    }

    /**
     * Validates the model for consistency and completeness.
     *
     * @return a ValidationResult indicating success or listing errors
     */
    public ValidationResult validate() {
        List<String> errors = new ArrayList<>();

        if (algorithmName == null || algorithmName.trim().isEmpty()) {
            errors.add("Algorithm name is required");
        }

        if (algorithmVersion < 1) {
            errors.add("Algorithm version must be positive");
        }

        if (modelId == null || modelId.trim().isEmpty()) {
            errors.add("Model ID is required");
        }

        // Validate matrix dimensions if present
        if (eigenVectorsData != null) {
            if (eigenVectorsData.length == 0) {
                errors.add("Eigenvectors matrix is empty");
            } else {
                int expectedCols = eigenVectorsData[0].length;
                for (int i = 1; i < eigenVectorsData.length; i++) {
                    if (eigenVectorsData[i].length != expectedCols) {
                        errors.add("Eigenvectors matrix has inconsistent column dimensions");
                        break;
                    }
                }
            }
        }

        // Validate mean face dimension matches eigenvectors
        if (meanFaceData != null && eigenVectorsData != null) {
            if (meanFaceData.length != eigenVectorsData.length) {
                errors.add("Mean face dimension does not match eigenvector rows");
            }
        }

        // Validate enrolled identities
        for (int i = 0; i < enrolledIdentities.size(); i++) {
            EnrolledIdentity ei = enrolledIdentities.get(i);
            if (ei.getIdentityId() == null || ei.getIdentityId().trim().isEmpty()) {
                errors.add("Enrolled identity at index " + i + " has no ID");
            }
            if (ei.getFeatureVectorData() == null || ei.getFeatureVectorData().length == 0) {
                errors.add("Enrolled identity " + ei.getIdentityId() + " has no feature vector");
            }
        }

        return new ValidationResult(errors.isEmpty(), errors);
    }

    /**
     * Creates a summary string of this model.
     *
     * @return a human-readable summary
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("TrainedModel Summary\n");
        sb.append("====================\n");
        sb.append(String.format("Algorithm: %s v%d\n", algorithmName, algorithmVersion));
        sb.append(String.format("Model ID: %s\n", modelId));
        sb.append(String.format("Format Version: %d\n", formatVersion));
        sb.append(String.format("Training Time: %s\n", getTrainingTimestamp()));

        if (eigenVectorsData != null) {
            sb.append(String.format("Eigenvectors: %d x %d\n",
                    eigenVectorsData.length, eigenVectorsData[0].length));
        }

        if (meanFaceData != null) {
            sb.append(String.format("Mean Face Dimension: %d\n", meanFaceData.length));
        }

        sb.append(String.format("Enrolled Identities: %d\n", enrolledIdentities.size()));
        sb.append(String.format("Hyperparameters: %d\n", hyperparameters.size()));

        if (metadata != null) {
            sb.append(String.format("Training Duration: %s\n", metadata.getTrainingDuration()));
            if (metadata.getAccuracy() != null) {
                sb.append(String.format("Accuracy: %.2f%%\n", metadata.getAccuracy() * 100));
            }
        }

        return sb.toString();
    }

    private double[][] copyArray(double[][] source) {
        double[][] copy = new double[source.length][];
        for (int i = 0; i < source.length; i++) {
            copy[i] = Arrays.copyOf(source[i], source[i].length);
        }
        return copy;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        TrainedModel that = (TrainedModel) o;
        return modelId.equals(that.modelId);
    }

    @Override
    public int hashCode() {
        return Objects.hash(modelId);
    }

    @Override
    public String toString() {
        return String.format("TrainedModel{algorithm='%s', version=%d, id='%s', identities=%d}",
                algorithmName, algorithmVersion, modelId.substring(0, 8), enrolledIdentities.size());
    }

    // ========================================================================
    // Inner Classes
    // ========================================================================

    /**
     * Represents an enrolled identity stored within the model.
     * Contains the identity's feature vector for recognition.
     */
    public static class EnrolledIdentity implements Serializable {
        private static final long serialVersionUID = 1L;

        private final String identityId;
        private final String identityName;
        private final double[] featureVectorData;
        private final String algorithmName;
        private final int algorithmVersion;
        private final long enrolledAtMillis;
        private final Map<String, String> metadata;

        /**
         * Creates an EnrolledIdentity from an Identity and its feature vector.
         *
         * @param identity the source identity
         * @param featureVector the computed feature vector
         */
        public EnrolledIdentity(Identity identity, FeatureVector featureVector) {
            this.identityId = identity.getId();
            this.identityName = identity.getName();
            this.featureVectorData = featureVector.getFeatures();
            this.algorithmName = featureVector.getAlgorithmName();
            this.algorithmVersion = featureVector.getAlgorithmVersion();
            this.enrolledAtMillis = System.currentTimeMillis();
            this.metadata = new HashMap<>(identity.getAllMetadata());
        }

        /**
         * Creates an EnrolledIdentity with explicit parameters.
         *
         * @param identityId the identity ID
         * @param identityName the identity name
         * @param featureVectorData the feature vector data
         * @param algorithmName the algorithm name
         * @param algorithmVersion the algorithm version
         */
        public EnrolledIdentity(String identityId, String identityName,
                                double[] featureVectorData, String algorithmName, int algorithmVersion) {
            this.identityId = identityId;
            this.identityName = identityName;
            this.featureVectorData = Arrays.copyOf(featureVectorData, featureVectorData.length);
            this.algorithmName = algorithmName;
            this.algorithmVersion = algorithmVersion;
            this.enrolledAtMillis = System.currentTimeMillis();
            this.metadata = new HashMap<>();
        }

        /**
         * Gets the identity ID.
         *
         * @return the identity ID
         */
        public String getIdentityId() {
            return identityId;
        }

        /**
         * Gets the identity name.
         *
         * @return the identity name
         */
        public String getIdentityName() {
            return identityName;
        }

        /**
         * Gets the feature vector data.
         *
         * @return a copy of the feature vector array
         */
        public double[] getFeatureVectorData() {
            return Arrays.copyOf(featureVectorData, featureVectorData.length);
        }

        /**
         * Gets the feature vector as a FeatureVector object.
         *
         * @return a new FeatureVector instance
         */
        public FeatureVector getFeatureVector() {
            return new FeatureVector(featureVectorData, algorithmName, algorithmVersion);
        }

        /**
         * Gets the algorithm name used for this feature vector.
         *
         * @return the algorithm name
         */
        public String getAlgorithmName() {
            return algorithmName;
        }

        /**
         * Gets the algorithm version.
         *
         * @return the algorithm version
         */
        public int getAlgorithmVersion() {
            return algorithmVersion;
        }

        /**
         * Gets the enrollment timestamp.
         *
         * @return the enrollment time
         */
        public LocalDateTime getEnrolledAt() {
            return LocalDateTime.ofInstant(
                    Instant.ofEpochMilli(enrolledAtMillis),
                    ZoneId.systemDefault());
        }

        /**
         * Gets the metadata map.
         *
         * @return unmodifiable metadata map
         */
        public Map<String, String> getMetadata() {
            return Collections.unmodifiableMap(metadata);
        }

        @Override
        public String toString() {
            return String.format("EnrolledIdentity{id='%s', name='%s', features=%d}",
                    identityId.substring(0, Math.min(8, identityId.length())),
                    identityName,
                    featureVectorData.length);
        }
    }

    /**
     * Metadata about the model including training metrics.
     */
    public static class ModelMetadata implements Serializable {
        private static final long serialVersionUID = 1L;

        private long trainingDurationMillis;
        private Double accuracy;
        private Double precision;
        private Double recall;
        private Double f1Score;
        private Integer trainingSetSize;
        private Integer validationSetSize;
        private String trainingEnvironment;
        private String notes;
        private final Map<String, Object> additionalMetrics;

        /**
         * Creates empty metadata.
         */
        public ModelMetadata() {
            this.additionalMetrics = new HashMap<>();
        }

        /**
         * Gets the training duration.
         *
         * @return the duration as a Duration object
         */
        public Duration getTrainingDuration() {
            return Duration.ofMillis(trainingDurationMillis);
        }

        /**
         * Sets the training duration.
         *
         * @param duration the training duration
         * @return this instance for chaining
         */
        public ModelMetadata setTrainingDuration(Duration duration) {
            this.trainingDurationMillis = duration.toMillis();
            return this;
        }

        /**
         * Sets the training duration in milliseconds.
         *
         * @param millis the duration in milliseconds
         * @return this instance for chaining
         */
        public ModelMetadata setTrainingDurationMillis(long millis) {
            this.trainingDurationMillis = millis;
            return this;
        }

        /**
         * Gets the accuracy metric.
         *
         * @return the accuracy (0.0-1.0), or null if not set
         */
        public Double getAccuracy() {
            return accuracy;
        }

        /**
         * Sets the accuracy metric.
         *
         * @param accuracy the accuracy (0.0-1.0)
         * @return this instance for chaining
         */
        public ModelMetadata setAccuracy(Double accuracy) {
            this.accuracy = accuracy;
            return this;
        }

        /**
         * Gets the precision metric.
         *
         * @return the precision, or null if not set
         */
        public Double getPrecision() {
            return precision;
        }

        /**
         * Sets the precision metric.
         *
         * @param precision the precision (0.0-1.0)
         * @return this instance for chaining
         */
        public ModelMetadata setPrecision(Double precision) {
            this.precision = precision;
            return this;
        }

        /**
         * Gets the recall metric.
         *
         * @return the recall, or null if not set
         */
        public Double getRecall() {
            return recall;
        }

        /**
         * Sets the recall metric.
         *
         * @param recall the recall (0.0-1.0)
         * @return this instance for chaining
         */
        public ModelMetadata setRecall(Double recall) {
            this.recall = recall;
            return this;
        }

        /**
         * Gets the F1 score.
         *
         * @return the F1 score, or null if not set
         */
        public Double getF1Score() {
            return f1Score;
        }

        /**
         * Sets the F1 score.
         *
         * @param f1Score the F1 score (0.0-1.0)
         * @return this instance for chaining
         */
        public ModelMetadata setF1Score(Double f1Score) {
            this.f1Score = f1Score;
            return this;
        }

        /**
         * Gets the training set size.
         *
         * @return the number of training samples, or null if not set
         */
        public Integer getTrainingSetSize() {
            return trainingSetSize;
        }

        /**
         * Sets the training set size.
         *
         * @param size the number of training samples
         * @return this instance for chaining
         */
        public ModelMetadata setTrainingSetSize(Integer size) {
            this.trainingSetSize = size;
            return this;
        }

        /**
         * Gets the validation set size.
         *
         * @return the number of validation samples, or null if not set
         */
        public Integer getValidationSetSize() {
            return validationSetSize;
        }

        /**
         * Sets the validation set size.
         *
         * @param size the number of validation samples
         * @return this instance for chaining
         */
        public ModelMetadata setValidationSetSize(Integer size) {
            this.validationSetSize = size;
            return this;
        }

        /**
         * Gets the training environment description.
         *
         * @return the environment description, or null if not set
         */
        public String getTrainingEnvironment() {
            return trainingEnvironment;
        }

        /**
         * Sets the training environment description.
         *
         * @param environment the environment description
         * @return this instance for chaining
         */
        public ModelMetadata setTrainingEnvironment(String environment) {
            this.trainingEnvironment = environment;
            return this;
        }

        /**
         * Gets any notes about the model.
         *
         * @return the notes, or null if not set
         */
        public String getNotes() {
            return notes;
        }

        /**
         * Sets notes about the model.
         *
         * @param notes the notes
         * @return this instance for chaining
         */
        public ModelMetadata setNotes(String notes) {
            this.notes = notes;
            return this;
        }

        /**
         * Sets a custom metric.
         *
         * @param name the metric name
         * @param value the metric value
         * @return this instance for chaining
         */
        public ModelMetadata setMetric(String name, Object value) {
            additionalMetrics.put(name, value);
            return this;
        }

        /**
         * Gets a custom metric.
         *
         * @param name the metric name
         * @param <T> the expected type
         * @return the metric value, or null if not set
         */
        @SuppressWarnings("unchecked")
        public <T> T getMetric(String name) {
            return (T) additionalMetrics.get(name);
        }

        /**
         * Gets all additional metrics.
         *
         * @return unmodifiable map of additional metrics
         */
        public Map<String, Object> getAdditionalMetrics() {
            return Collections.unmodifiableMap(additionalMetrics);
        }

        @Override
        public String toString() {
            return String.format("ModelMetadata{duration=%s, accuracy=%s, trainSize=%s}",
                    getTrainingDuration(),
                    accuracy != null ? String.format("%.2f%%", accuracy * 100) : "N/A",
                    trainingSetSize != null ? trainingSetSize : "N/A");
        }
    }

    /**
     * Result of model validation.
     */
    public static class ValidationResult implements Serializable {
        private static final long serialVersionUID = 1L;

        private final boolean valid;
        private final List<String> errors;

        /**
         * Creates a validation result.
         *
         * @param valid whether validation passed
         * @param errors list of validation errors
         */
        public ValidationResult(boolean valid, List<String> errors) {
            this.valid = valid;
            this.errors = errors != null ? new ArrayList<>(errors) : Collections.emptyList();
        }

        /**
         * Checks if the model is valid.
         *
         * @return true if validation passed
         */
        public boolean isValid() {
            return valid;
        }

        /**
         * Gets the list of validation errors.
         *
         * @return unmodifiable list of errors
         */
        public List<String> getErrors() {
            return Collections.unmodifiableList(errors);
        }

        /**
         * Gets validation errors as a single string.
         *
         * @return newline-separated error messages
         */
        public String getErrorsAsString() {
            return String.join("\n", errors);
        }

        @Override
        public String toString() {
            return valid ? "ValidationResult{valid=true}"
                    : String.format("ValidationResult{valid=false, errors=%d}", errors.size());
        }
    }

    /**
     * Builder for creating TrainedModel instances.
     */
    public static class Builder {
        private final String algorithmName;
        private final int algorithmVersion;
        private String modelId;
        private LocalDateTime trainingTimestamp;
        private double[][] eigenVectorsData;
        private double[][] eigenValuesData;
        private double[] meanFaceData;
        private final List<EnrolledIdentity> enrolledIdentities = new ArrayList<>();
        private final Map<String, Object> hyperparameters = new HashMap<>();
        private final Map<String, Object> configuration = new HashMap<>();
        private ModelMetadata metadata;

        /**
         * Creates a new builder.
         *
         * @param algorithmName the algorithm name
         * @param algorithmVersion the algorithm version
         */
        private Builder(String algorithmName, int algorithmVersion) {
            this.algorithmName = Objects.requireNonNull(algorithmName, "Algorithm name cannot be null");
            this.algorithmVersion = algorithmVersion;
        }

        /**
         * Sets the model ID.
         *
         * @param modelId the model ID
         * @return this builder
         */
        public Builder modelId(String modelId) {
            this.modelId = modelId;
            return this;
        }

        /**
         * Sets the training timestamp.
         *
         * @param timestamp the training timestamp
         * @return this builder
         */
        public Builder trainingTimestamp(LocalDateTime timestamp) {
            this.trainingTimestamp = timestamp;
            return this;
        }

        /**
         * Sets the eigenvectors from a JAMA Matrix.
         *
         * @param eigenVectors the eigenvector matrix
         * @return this builder
         */
        public Builder eigenVectors(Matrix eigenVectors) {
            this.eigenVectorsData = eigenVectors != null ? eigenVectors.getArray() : null;
            return this;
        }

        /**
         * Sets the eigenvectors from a 2D array.
         *
         * @param eigenVectorsData the eigenvector data
         * @return this builder
         */
        public Builder eigenVectorsData(double[][] eigenVectorsData) {
            this.eigenVectorsData = eigenVectorsData;
            return this;
        }

        /**
         * Sets the eigenvalues from a JAMA Matrix.
         *
         * @param eigenValues the eigenvalue matrix (diagonal)
         * @return this builder
         */
        public Builder eigenValues(Matrix eigenValues) {
            this.eigenValuesData = eigenValues != null ? eigenValues.getArray() : null;
            return this;
        }

        /**
         * Sets the eigenvalues from a 2D array.
         *
         * @param eigenValuesData the eigenvalue data
         * @return this builder
         */
        public Builder eigenValuesData(double[][] eigenValuesData) {
            this.eigenValuesData = eigenValuesData;
            return this;
        }

        /**
         * Sets the mean face from a JAMA Matrix.
         *
         * @param meanFace the mean face matrix (column vector)
         * @return this builder
         */
        public Builder meanFace(Matrix meanFace) {
            this.meanFaceData = meanFace != null ? meanFace.getColumnPackedCopy() : null;
            return this;
        }

        /**
         * Sets the mean face from an array.
         *
         * @param meanFaceData the mean face data
         * @return this builder
         */
        public Builder meanFaceData(double[] meanFaceData) {
            this.meanFaceData = meanFaceData;
            return this;
        }

        /**
         * Adds an enrolled identity.
         *
         * @param identity the identity to add
         * @param featureVector the identity's feature vector
         * @return this builder
         */
        public Builder addIdentity(Identity identity, FeatureVector featureVector) {
            this.enrolledIdentities.add(new EnrolledIdentity(identity, featureVector));
            return this;
        }

        /**
         * Adds an enrolled identity.
         *
         * @param enrolledIdentity the enrolled identity to add
         * @return this builder
         */
        public Builder addIdentity(EnrolledIdentity enrolledIdentity) {
            this.enrolledIdentities.add(enrolledIdentity);
            return this;
        }

        /**
         * Sets all enrolled identities.
         *
         * @param identities the list of enrolled identities
         * @return this builder
         */
        public Builder enrolledIdentities(List<EnrolledIdentity> identities) {
            this.enrolledIdentities.clear();
            this.enrolledIdentities.addAll(identities);
            return this;
        }

        /**
         * Sets a hyperparameter.
         *
         * @param name the hyperparameter name
         * @param value the hyperparameter value
         * @return this builder
         */
        public Builder setHyperparameter(String name, Object value) {
            this.hyperparameters.put(name, value);
            return this;
        }

        /**
         * Sets all hyperparameters.
         *
         * @param hyperparameters the hyperparameters map
         * @return this builder
         */
        public Builder hyperparameters(Map<String, Object> hyperparameters) {
            this.hyperparameters.clear();
            this.hyperparameters.putAll(hyperparameters);
            return this;
        }

        /**
         * Sets a configuration value.
         *
         * @param name the configuration name
         * @param value the configuration value
         * @return this builder
         */
        public Builder setConfig(String name, Object value) {
            this.configuration.put(name, value);
            return this;
        }

        /**
         * Sets all configuration values.
         *
         * @param configuration the configuration map
         * @return this builder
         */
        public Builder configuration(Map<String, Object> configuration) {
            this.configuration.clear();
            this.configuration.putAll(configuration);
            return this;
        }

        /**
         * Sets the model metadata.
         *
         * @param metadata the metadata
         * @return this builder
         */
        public Builder metadata(ModelMetadata metadata) {
            this.metadata = metadata;
            return this;
        }

        /**
         * Builds the TrainedModel instance.
         *
         * @return a new TrainedModel
         */
        public TrainedModel build() {
            return new TrainedModel(this);
        }
    }
}
