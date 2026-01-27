package com.facerecognition.infrastructure.persistence;

import java.io.IOException;
import java.nio.file.Path;
import java.time.LocalDateTime;
import java.util.List;
import java.util.Optional;

/**
 * Repository interface for persisting and retrieving trained face recognition models.
 *
 * <p>This interface defines the contract for model storage, supporting operations
 * for saving, loading, listing, and managing trained models. Implementations may
 * use various storage backends such as:</p>
 *
 * <ul>
 *   <li>Local file system (see {@link FileModelRepository})</li>
 *   <li>Cloud storage (S3, GCS, Azure Blob)</li>
 *   <li>Database (relational or document)</li>
 *   <li>Distributed cache</li>
 * </ul>
 *
 * <h3>Model Versioning:</h3>
 * <p>The repository supports model versioning, allowing multiple versions of a
 * model to be stored and retrieved. This is useful for:</p>
 * <ul>
 *   <li>A/B testing different model versions</li>
 *   <li>Rollback to previous versions</li>
 *   <li>Gradual model updates</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * ModelRepository repository = new FileModelRepository(Paths.get("/models"));
 *
 * // Save a model
 * repository.save(trainedModel, "face-recognition-v1");
 *
 * // Load the model
 * Optional<TrainedModel> loaded = repository.load("face-recognition-v1");
 *
 * // List all models
 * List<ModelInfo> models = repository.listModels();
 *
 * // Delete an old model
 * repository.delete("old-model");
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see TrainedModel
 * @see FileModelRepository
 */
public interface ModelRepository {

    /**
     * Saves a trained model with the specified name.
     *
     * <p>If a model with the same name already exists, the behavior depends
     * on the implementation. By default, it should overwrite the existing model.
     * Use {@link #save(TrainedModel, String, boolean)} for explicit control.</p>
     *
     * @param model the trained model to save
     * @param modelName the unique name for this model
     * @throws IOException if saving fails due to I/O errors
     * @throws IllegalArgumentException if model or name is null/invalid
     * @throws ModelPersistenceException if model validation fails
     */
    void save(TrainedModel model, String modelName) throws IOException;

    /**
     * Saves a trained model with explicit overwrite control.
     *
     * @param model the trained model to save
     * @param modelName the unique name for this model
     * @param overwrite true to overwrite existing model, false to throw exception
     * @throws IOException if saving fails due to I/O errors
     * @throws ModelPersistenceException if model exists and overwrite is false
     */
    void save(TrainedModel model, String modelName, boolean overwrite) throws IOException;

    /**
     * Loads a trained model by name.
     *
     * @param modelName the name of the model to load
     * @return an Optional containing the model if found, empty otherwise
     * @throws IOException if loading fails due to I/O errors
     * @throws ModelPersistenceException if loaded model fails validation
     */
    Optional<TrainedModel> load(String modelName) throws IOException;

    /**
     * Loads a trained model with validation options.
     *
     * @param modelName the name of the model to load
     * @param validate whether to validate the model after loading
     * @return an Optional containing the model if found, empty otherwise
     * @throws IOException if loading fails due to I/O errors
     * @throws ModelPersistenceException if validation is requested and fails
     */
    Optional<TrainedModel> load(String modelName, boolean validate) throws IOException;

    /**
     * Checks if a model with the given name exists.
     *
     * @param modelName the model name to check
     * @return true if the model exists
     */
    boolean exists(String modelName);

    /**
     * Deletes a model by name.
     *
     * @param modelName the name of the model to delete
     * @return true if the model was deleted, false if it didn't exist
     * @throws IOException if deletion fails due to I/O errors
     */
    boolean delete(String modelName) throws IOException;

    /**
     * Lists all available models.
     *
     * @return list of model information objects
     * @throws IOException if listing fails
     */
    List<ModelInfo> listModels() throws IOException;

    /**
     * Lists models filtered by algorithm.
     *
     * @param algorithmName the algorithm name to filter by
     * @return list of matching model information objects
     * @throws IOException if listing fails
     */
    List<ModelInfo> listModelsByAlgorithm(String algorithmName) throws IOException;

    /**
     * Gets information about a specific model without loading it.
     *
     * @param modelName the model name
     * @return an Optional containing model info if found
     * @throws IOException if reading metadata fails
     */
    Optional<ModelInfo> getModelInfo(String modelName) throws IOException;

    /**
     * Exports a model to a specific path.
     *
     * @param modelName the model to export
     * @param destination the destination path
     * @throws IOException if export fails
     */
    void export(String modelName, Path destination) throws IOException;

    /**
     * Imports a model from a file.
     *
     * @param source the source file path
     * @param modelName the name to give the imported model
     * @return information about the imported model
     * @throws IOException if import fails
     * @throws ModelPersistenceException if the file is not a valid model
     */
    ModelInfo importModel(Path source, String modelName) throws IOException;

    /**
     * Creates a backup of a model.
     *
     * @param modelName the model to backup
     * @param backupName the name for the backup
     * @throws IOException if backup fails
     */
    void backup(String modelName, String backupName) throws IOException;

    /**
     * Restores a model from a backup.
     *
     * @param backupName the backup to restore
     * @param modelName the name for the restored model
     * @throws IOException if restore fails
     */
    void restore(String backupName, String modelName) throws IOException;

    /**
     * Gets the storage location/path for models.
     *
     * @return the storage path
     */
    Path getStoragePath();

    /**
     * Clears the model cache if the implementation uses caching.
     */
    default void clearCache() {
        // Default no-op for implementations without caching
    }

    /**
     * Gets the total storage size used by all models.
     *
     * @return the total size in bytes
     * @throws IOException if calculation fails
     */
    long getTotalStorageSize() throws IOException;

    // ========================================================================
    // Inner Classes
    // ========================================================================

    /**
     * Information about a stored model without loading the full model.
     */
    class ModelInfo {
        private final String modelName;
        private final String algorithmName;
        private final int algorithmVersion;
        private final String modelId;
        private final LocalDateTime trainingTimestamp;
        private final LocalDateTime savedTimestamp;
        private final long fileSizeBytes;
        private final int identityCount;
        private final int formatVersion;
        private final String checksum;
        private final boolean compressed;
        private final Path filePath;

        /**
         * Creates a ModelInfo instance.
         *
         * @param builder the builder containing model information
         */
        private ModelInfo(Builder builder) {
            this.modelName = builder.modelName;
            this.algorithmName = builder.algorithmName;
            this.algorithmVersion = builder.algorithmVersion;
            this.modelId = builder.modelId;
            this.trainingTimestamp = builder.trainingTimestamp;
            this.savedTimestamp = builder.savedTimestamp;
            this.fileSizeBytes = builder.fileSizeBytes;
            this.identityCount = builder.identityCount;
            this.formatVersion = builder.formatVersion;
            this.checksum = builder.checksum;
            this.compressed = builder.compressed;
            this.filePath = builder.filePath;
        }

        /**
         * Creates a new builder for ModelInfo.
         *
         * @return a new Builder instance
         */
        public static Builder builder() {
            return new Builder();
        }

        /**
         * Gets the model name.
         *
         * @return the model name
         */
        public String getModelName() {
            return modelName;
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
         * @return the algorithm version
         */
        public int getAlgorithmVersion() {
            return algorithmVersion;
        }

        /**
         * Gets the model ID.
         *
         * @return the model ID
         */
        public String getModelId() {
            return modelId;
        }

        /**
         * Gets the training timestamp.
         *
         * @return the training timestamp
         */
        public LocalDateTime getTrainingTimestamp() {
            return trainingTimestamp;
        }

        /**
         * Gets the timestamp when the model was saved.
         *
         * @return the save timestamp
         */
        public LocalDateTime getSavedTimestamp() {
            return savedTimestamp;
        }

        /**
         * Gets the file size in bytes.
         *
         * @return the file size
         */
        public long getFileSizeBytes() {
            return fileSizeBytes;
        }

        /**
         * Gets the number of enrolled identities.
         *
         * @return the identity count
         */
        public int getIdentityCount() {
            return identityCount;
        }

        /**
         * Gets the format version.
         *
         * @return the format version
         */
        public int getFormatVersion() {
            return formatVersion;
        }

        /**
         * Gets the checksum of the model file.
         *
         * @return the checksum string
         */
        public String getChecksum() {
            return checksum;
        }

        /**
         * Checks if the model is compressed.
         *
         * @return true if compressed
         */
        public boolean isCompressed() {
            return compressed;
        }

        /**
         * Gets the file path if available.
         *
         * @return the file path, may be null for non-file implementations
         */
        public Path getFilePath() {
            return filePath;
        }

        /**
         * Gets a human-readable file size.
         *
         * @return formatted file size string
         */
        public String getFormattedFileSize() {
            if (fileSizeBytes < 1024) {
                return fileSizeBytes + " B";
            } else if (fileSizeBytes < 1024 * 1024) {
                return String.format("%.1f KB", fileSizeBytes / 1024.0);
            } else if (fileSizeBytes < 1024 * 1024 * 1024) {
                return String.format("%.1f MB", fileSizeBytes / (1024.0 * 1024.0));
            } else {
                return String.format("%.1f GB", fileSizeBytes / (1024.0 * 1024.0 * 1024.0));
            }
        }

        @Override
        public String toString() {
            return String.format("ModelInfo{name='%s', algorithm='%s v%d', identities=%d, size=%s}",
                    modelName, algorithmName, algorithmVersion, identityCount, getFormattedFileSize());
        }

        /**
         * Builder for ModelInfo.
         */
        public static class Builder {
            private String modelName;
            private String algorithmName;
            private int algorithmVersion;
            private String modelId;
            private LocalDateTime trainingTimestamp;
            private LocalDateTime savedTimestamp;
            private long fileSizeBytes;
            private int identityCount;
            private int formatVersion;
            private String checksum;
            private boolean compressed;
            private Path filePath;

            /**
             * Sets the model name.
             *
             * @param modelName the model name
             * @return this builder
             */
            public Builder modelName(String modelName) {
                this.modelName = modelName;
                return this;
            }

            /**
             * Sets the algorithm name.
             *
             * @param algorithmName the algorithm name
             * @return this builder
             */
            public Builder algorithmName(String algorithmName) {
                this.algorithmName = algorithmName;
                return this;
            }

            /**
             * Sets the algorithm version.
             *
             * @param algorithmVersion the algorithm version
             * @return this builder
             */
            public Builder algorithmVersion(int algorithmVersion) {
                this.algorithmVersion = algorithmVersion;
                return this;
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
             * @param trainingTimestamp the training timestamp
             * @return this builder
             */
            public Builder trainingTimestamp(LocalDateTime trainingTimestamp) {
                this.trainingTimestamp = trainingTimestamp;
                return this;
            }

            /**
             * Sets the saved timestamp.
             *
             * @param savedTimestamp the saved timestamp
             * @return this builder
             */
            public Builder savedTimestamp(LocalDateTime savedTimestamp) {
                this.savedTimestamp = savedTimestamp;
                return this;
            }

            /**
             * Sets the file size in bytes.
             *
             * @param fileSizeBytes the file size
             * @return this builder
             */
            public Builder fileSizeBytes(long fileSizeBytes) {
                this.fileSizeBytes = fileSizeBytes;
                return this;
            }

            /**
             * Sets the identity count.
             *
             * @param identityCount the number of enrolled identities
             * @return this builder
             */
            public Builder identityCount(int identityCount) {
                this.identityCount = identityCount;
                return this;
            }

            /**
             * Sets the format version.
             *
             * @param formatVersion the format version
             * @return this builder
             */
            public Builder formatVersion(int formatVersion) {
                this.formatVersion = formatVersion;
                return this;
            }

            /**
             * Sets the checksum.
             *
             * @param checksum the checksum string
             * @return this builder
             */
            public Builder checksum(String checksum) {
                this.checksum = checksum;
                return this;
            }

            /**
             * Sets whether the model is compressed.
             *
             * @param compressed true if compressed
             * @return this builder
             */
            public Builder compressed(boolean compressed) {
                this.compressed = compressed;
                return this;
            }

            /**
             * Sets the file path.
             *
             * @param filePath the file path
             * @return this builder
             */
            public Builder filePath(Path filePath) {
                this.filePath = filePath;
                return this;
            }

            /**
             * Builds the ModelInfo instance.
             *
             * @return a new ModelInfo
             */
            public ModelInfo build() {
                return new ModelInfo(this);
            }
        }
    }

    /**
     * Exception thrown when model persistence operations fail.
     */
    class ModelPersistenceException extends RuntimeException {
        private static final long serialVersionUID = 1L;

        private final String modelName;
        private final ErrorType errorType;

        /**
         * Creates a new exception.
         *
         * @param message the error message
         * @param modelName the model name involved
         * @param errorType the type of error
         */
        public ModelPersistenceException(String message, String modelName, ErrorType errorType) {
            super(message);
            this.modelName = modelName;
            this.errorType = errorType;
        }

        /**
         * Creates a new exception with a cause.
         *
         * @param message the error message
         * @param cause the underlying cause
         * @param modelName the model name involved
         * @param errorType the type of error
         */
        public ModelPersistenceException(String message, Throwable cause, String modelName, ErrorType errorType) {
            super(message, cause);
            this.modelName = modelName;
            this.errorType = errorType;
        }

        /**
         * Gets the model name involved in the error.
         *
         * @return the model name
         */
        public String getModelName() {
            return modelName;
        }

        /**
         * Gets the error type.
         *
         * @return the error type
         */
        public ErrorType getErrorType() {
            return errorType;
        }

        /**
         * Types of persistence errors.
         */
        public enum ErrorType {
            /** Model already exists when trying to save without overwrite */
            MODEL_ALREADY_EXISTS,
            /** Model not found when trying to load */
            MODEL_NOT_FOUND,
            /** Model file is corrupted */
            CORRUPTED_MODEL,
            /** Model validation failed */
            VALIDATION_FAILED,
            /** Checksum mismatch */
            CHECKSUM_MISMATCH,
            /** Incompatible format version */
            INCOMPATIBLE_VERSION,
            /** I/O error */
            IO_ERROR,
            /** Serialization error */
            SERIALIZATION_ERROR,
            /** Permission denied */
            PERMISSION_DENIED,
            /** Storage full */
            STORAGE_FULL,
            /** Unknown error */
            UNKNOWN
        }
    }
}
