package com.facerecognition.infrastructure.persistence;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * File-based implementation of the ModelRepository interface.
 *
 * <p>This implementation stores trained face recognition models as binary files
 * with GZIP compression, alongside JSON metadata files for quick model discovery
 * without full deserialization.</p>
 *
 * <h3>Storage Structure:</h3>
 * <pre>
 * storage_path/
 *   models/
 *     model-name.frm         (compressed binary model)
 *     model-name.json        (JSON metadata)
 *   backups/
 *     backup-name.frm
 *     backup-name.json
 * </pre>
 *
 * <h3>Features:</h3>
 * <ul>
 *   <li>GZIP compression for efficient storage</li>
 *   <li>JSON metadata for quick model discovery</li>
 *   <li>Model versioning and backward compatibility</li>
 *   <li>Checksum validation on load</li>
 *   <li>Thread-safe operations with read-write locks</li>
 *   <li>Optional in-memory caching</li>
 *   <li>Atomic file operations for safety</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * // Create repository
 * FileModelRepository repository = new FileModelRepository(Paths.get("/var/models"));
 *
 * // Save a model
 * repository.save(trainedModel, "production-v1");
 *
 * // Load with validation
 * TrainedModel loaded = repository.load("production-v1", true)
 *     .orElseThrow(() -> new RuntimeException("Model not found"));
 *
 * // List all models
 * List<ModelInfo> models = repository.listModels();
 * models.forEach(System.out::println);
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see ModelRepository
 * @see TrainedModel
 * @see ModelSerializer
 */
public class FileModelRepository implements ModelRepository {

    private static final Logger logger = LoggerFactory.getLogger(FileModelRepository.class);

    private static final String MODELS_DIR = "models";
    private static final String BACKUPS_DIR = "backups";

    private final Path storagePath;
    private final Path modelsPath;
    private final Path backupsPath;
    private final ObjectMapper objectMapper;
    private final boolean compressionEnabled;
    private final boolean cachingEnabled;
    private final boolean validateOnLoad;

    // Thread-safety
    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    private final Map<String, TrainedModel> modelCache = new ConcurrentHashMap<>();

    /**
     * Creates a FileModelRepository with default settings.
     *
     * @param storagePath the root directory for model storage
     * @throws IOException if the directory cannot be created
     */
    public FileModelRepository(Path storagePath) throws IOException {
        this(storagePath, new Config());
    }

    /**
     * Creates a FileModelRepository with custom configuration.
     *
     * @param storagePath the root directory for model storage
     * @param config the repository configuration
     * @throws IOException if the directory cannot be created
     */
    public FileModelRepository(Path storagePath, Config config) throws IOException {
        this.storagePath = Objects.requireNonNull(storagePath, "Storage path cannot be null").toAbsolutePath();
        this.modelsPath = this.storagePath.resolve(MODELS_DIR);
        this.backupsPath = this.storagePath.resolve(BACKUPS_DIR);
        this.compressionEnabled = config.compressionEnabled;
        this.cachingEnabled = config.cachingEnabled;
        this.validateOnLoad = config.validateOnLoad;

        // Initialize ObjectMapper for JSON metadata
        this.objectMapper = new ObjectMapper();
        this.objectMapper.registerModule(new JavaTimeModule());
        this.objectMapper.enable(SerializationFeature.INDENT_OUTPUT);
        this.objectMapper.disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);

        // Create directory structure
        initializeStorage();

        logger.info("FileModelRepository initialized at: {}", storagePath);
    }

    private void initializeStorage() throws IOException {
        try {
            Files.createDirectories(modelsPath);
            Files.createDirectories(backupsPath);
        } catch (IOException e) {
            throw new IOException("Failed to initialize storage directories at: " + storagePath, e);
        }
    }

    @Override
    public void save(TrainedModel model, String modelName) throws IOException {
        save(model, modelName, true);
    }

    @Override
    public void save(TrainedModel model, String modelName, boolean overwrite) throws IOException {
        Objects.requireNonNull(model, "Model cannot be null");
        validateModelName(modelName);

        lock.writeLock().lock();
        try {
            Path modelFile = getModelPath(modelName);
            Path metadataFile = getMetadataPath(modelName);

            // Check if model exists
            if (Files.exists(modelFile) && !overwrite) {
                throw new ModelPersistenceException(
                        "Model already exists: " + modelName,
                        modelName,
                        ModelPersistenceException.ErrorType.MODEL_ALREADY_EXISTS);
            }

            // Validate model before saving
            TrainedModel.ValidationResult validation = model.validate();
            if (!validation.isValid()) {
                throw new ModelPersistenceException(
                        "Model validation failed: " + validation.getErrorsAsString(),
                        modelName,
                        ModelPersistenceException.ErrorType.VALIDATION_FAILED);
            }

            // Serialize model to temporary file first (atomic operation)
            Path tempModelFile = Files.createTempFile(modelsPath, "model-", ".tmp");
            Path tempMetadataFile = Files.createTempFile(modelsPath, "metadata-", ".tmp");

            try {
                // Serialize model
                logger.debug("Serializing model '{}' with compression: {}", modelName, compressionEnabled);
                ModelSerializer.serializeToFile(model, tempModelFile, compressionEnabled);

                // Create and write metadata
                ModelMetadataJson metadata = createMetadata(model, modelName, tempModelFile);
                objectMapper.writeValue(tempMetadataFile.toFile(), metadata);

                // Atomic move (replace existing if overwrite is true)
                Files.move(tempModelFile, modelFile, StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);
                Files.move(tempMetadataFile, metadataFile, StandardCopyOption.REPLACE_EXISTING,
                        StandardCopyOption.ATOMIC_MOVE);

                // Update cache
                if (cachingEnabled) {
                    modelCache.put(modelName, model);
                }

                logger.info("Model '{}' saved successfully ({} bytes)", modelName, Files.size(modelFile));

            } catch (Exception e) {
                // Cleanup temp files on failure
                Files.deleteIfExists(tempModelFile);
                Files.deleteIfExists(tempMetadataFile);
                throw e;
            }

        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public Optional<TrainedModel> load(String modelName) throws IOException {
        return load(modelName, validateOnLoad);
    }

    @Override
    public Optional<TrainedModel> load(String modelName, boolean validate) throws IOException {
        validateModelName(modelName);

        // Check cache first
        if (cachingEnabled && modelCache.containsKey(modelName)) {
            logger.debug("Returning cached model: {}", modelName);
            return Optional.of(modelCache.get(modelName));
        }

        lock.readLock().lock();
        try {
            Path modelFile = getModelPath(modelName);

            if (!Files.exists(modelFile)) {
                return Optional.empty();
            }

            logger.debug("Loading model '{}' from file", modelName);

            // Validate checksum before loading
            if (!ModelSerializer.validateChecksum(modelFile)) {
                throw new ModelPersistenceException(
                        "Model file checksum validation failed: " + modelName,
                        modelName,
                        ModelPersistenceException.ErrorType.CHECKSUM_MISMATCH);
            }

            // Deserialize model
            TrainedModel model = ModelSerializer.deserializeFromFile(modelFile);

            // Validate model structure if requested
            if (validate) {
                TrainedModel.ValidationResult validation = model.validate();
                if (!validation.isValid()) {
                    throw new ModelPersistenceException(
                            "Loaded model validation failed: " + validation.getErrorsAsString(),
                            modelName,
                            ModelPersistenceException.ErrorType.VALIDATION_FAILED);
                }
            }

            // Update cache
            if (cachingEnabled) {
                modelCache.put(modelName, model);
            }

            logger.info("Model '{}' loaded successfully", modelName);
            return Optional.of(model);

        } catch (ModelSerializer.SerializationException e) {
            throw new ModelPersistenceException(
                    "Failed to deserialize model: " + e.getMessage(),
                    e,
                    modelName,
                    ModelPersistenceException.ErrorType.CORRUPTED_MODEL);
        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public boolean exists(String modelName) {
        validateModelName(modelName);
        return Files.exists(getModelPath(modelName));
    }

    @Override
    public boolean delete(String modelName) throws IOException {
        validateModelName(modelName);

        lock.writeLock().lock();
        try {
            Path modelFile = getModelPath(modelName);
            Path metadataFile = getMetadataPath(modelName);

            if (!Files.exists(modelFile)) {
                return false;
            }

            Files.deleteIfExists(modelFile);
            Files.deleteIfExists(metadataFile);

            // Remove from cache
            modelCache.remove(modelName);

            logger.info("Model '{}' deleted", modelName);
            return true;

        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public List<ModelInfo> listModels() throws IOException {
        lock.readLock().lock();
        try {
            List<ModelInfo> models = new ArrayList<>();

            try (Stream<Path> files = Files.list(modelsPath)) {
                List<Path> modelFiles = files
                        .filter(p -> p.toString().endsWith(ModelSerializer.MODEL_EXTENSION))
                        .collect(Collectors.toList());

                for (Path modelFile : modelFiles) {
                    try {
                        String modelName = extractModelName(modelFile);
                        ModelInfo info = buildModelInfo(modelName, modelFile);
                        models.add(info);
                    } catch (Exception e) {
                        logger.warn("Failed to read model info for: {}", modelFile, e);
                    }
                }
            }

            // Sort by saved timestamp (newest first)
            models.sort((a, b) -> {
                if (a.getSavedTimestamp() == null) return 1;
                if (b.getSavedTimestamp() == null) return -1;
                return b.getSavedTimestamp().compareTo(a.getSavedTimestamp());
            });

            return models;

        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public List<ModelInfo> listModelsByAlgorithm(String algorithmName) throws IOException {
        return listModels().stream()
                .filter(m -> algorithmName.equals(m.getAlgorithmName()))
                .collect(Collectors.toList());
    }

    @Override
    public Optional<ModelInfo> getModelInfo(String modelName) throws IOException {
        validateModelName(modelName);

        lock.readLock().lock();
        try {
            Path modelFile = getModelPath(modelName);

            if (!Files.exists(modelFile)) {
                return Optional.empty();
            }

            return Optional.of(buildModelInfo(modelName, modelFile));

        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public void export(String modelName, Path destination) throws IOException {
        validateModelName(modelName);
        Objects.requireNonNull(destination, "Destination cannot be null");

        lock.readLock().lock();
        try {
            Path modelFile = getModelPath(modelName);

            if (!Files.exists(modelFile)) {
                throw new ModelPersistenceException(
                        "Model not found: " + modelName,
                        modelName,
                        ModelPersistenceException.ErrorType.MODEL_NOT_FOUND);
            }

            // Create parent directories if needed
            Files.createDirectories(destination.getParent());

            // Copy the model file
            Files.copy(modelFile, destination, StandardCopyOption.REPLACE_EXISTING);

            // Also copy metadata if it exists
            Path metadataFile = getMetadataPath(modelName);
            if (Files.exists(metadataFile)) {
                Path metadataDest = destination.resolveSibling(
                        destination.getFileName().toString().replace(
                                ModelSerializer.MODEL_EXTENSION,
                                ModelSerializer.METADATA_EXTENSION));
                Files.copy(metadataFile, metadataDest, StandardCopyOption.REPLACE_EXISTING);
            }

            logger.info("Model '{}' exported to: {}", modelName, destination);

        } finally {
            lock.readLock().unlock();
        }
    }

    @Override
    public ModelInfo importModel(Path source, String modelName) throws IOException {
        Objects.requireNonNull(source, "Source cannot be null");
        validateModelName(modelName);

        if (!Files.exists(source)) {
            throw new IOException("Source file does not exist: " + source);
        }

        // Verify it's a valid model file
        if (!ModelSerializer.isModelFile(source)) {
            throw new ModelPersistenceException(
                    "Not a valid model file: " + source,
                    modelName,
                    ModelPersistenceException.ErrorType.CORRUPTED_MODEL);
        }

        // Load and save to perform validation and create metadata
        TrainedModel model = ModelSerializer.deserializeFromFile(source);
        save(model, modelName);

        return getModelInfo(modelName).orElseThrow();
    }

    @Override
    public void backup(String modelName, String backupName) throws IOException {
        validateModelName(modelName);
        validateModelName(backupName);

        lock.writeLock().lock();
        try {
            Path modelFile = getModelPath(modelName);
            Path metadataFile = getMetadataPath(modelName);

            if (!Files.exists(modelFile)) {
                throw new ModelPersistenceException(
                        "Model not found: " + modelName,
                        modelName,
                        ModelPersistenceException.ErrorType.MODEL_NOT_FOUND);
            }

            Path backupModelFile = backupsPath.resolve(backupName + ModelSerializer.MODEL_EXTENSION);
            Path backupMetadataFile = backupsPath.resolve(backupName + ModelSerializer.METADATA_EXTENSION);

            Files.copy(modelFile, backupModelFile, StandardCopyOption.REPLACE_EXISTING);
            if (Files.exists(metadataFile)) {
                Files.copy(metadataFile, backupMetadataFile, StandardCopyOption.REPLACE_EXISTING);
            }

            logger.info("Model '{}' backed up as '{}'", modelName, backupName);

        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public void restore(String backupName, String modelName) throws IOException {
        validateModelName(backupName);
        validateModelName(modelName);

        lock.writeLock().lock();
        try {
            Path backupModelFile = backupsPath.resolve(backupName + ModelSerializer.MODEL_EXTENSION);

            if (!Files.exists(backupModelFile)) {
                throw new ModelPersistenceException(
                        "Backup not found: " + backupName,
                        backupName,
                        ModelPersistenceException.ErrorType.MODEL_NOT_FOUND);
            }

            Path modelFile = getModelPath(modelName);
            Path metadataFile = getMetadataPath(modelName);

            Files.copy(backupModelFile, modelFile, StandardCopyOption.REPLACE_EXISTING);

            Path backupMetadataFile = backupsPath.resolve(backupName + ModelSerializer.METADATA_EXTENSION);
            if (Files.exists(backupMetadataFile)) {
                Files.copy(backupMetadataFile, metadataFile, StandardCopyOption.REPLACE_EXISTING);
            }

            // Invalidate cache
            modelCache.remove(modelName);

            logger.info("Backup '{}' restored as model '{}'", backupName, modelName);

        } finally {
            lock.writeLock().unlock();
        }
    }

    @Override
    public Path getStoragePath() {
        return storagePath;
    }

    @Override
    public void clearCache() {
        modelCache.clear();
        logger.debug("Model cache cleared");
    }

    @Override
    public long getTotalStorageSize() throws IOException {
        lock.readLock().lock();
        try {
            final long[] totalSize = {0};

            Files.walkFileTree(storagePath, new SimpleFileVisitor<>() {
                @Override
                public FileVisitResult visitFile(Path file, BasicFileAttributes attrs) {
                    totalSize[0] += attrs.size();
                    return FileVisitResult.CONTINUE;
                }
            });

            return totalSize[0];

        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Gets the number of models in the repository.
     *
     * @return the model count
     * @throws IOException if listing fails
     */
    public int getModelCount() throws IOException {
        return listModels().size();
    }

    /**
     * Gets the number of backups in the repository.
     *
     * @return the backup count
     * @throws IOException if listing fails
     */
    public int getBackupCount() throws IOException {
        lock.readLock().lock();
        try (Stream<Path> files = Files.list(backupsPath)) {
            return (int) files
                    .filter(p -> p.toString().endsWith(ModelSerializer.MODEL_EXTENSION))
                    .count();
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Lists all backups.
     *
     * @return list of backup names
     * @throws IOException if listing fails
     */
    public List<String> listBackups() throws IOException {
        lock.readLock().lock();
        try (Stream<Path> files = Files.list(backupsPath)) {
            return files
                    .filter(p -> p.toString().endsWith(ModelSerializer.MODEL_EXTENSION))
                    .map(this::extractModelName)
                    .collect(Collectors.toList());
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Deletes a backup.
     *
     * @param backupName the backup name
     * @return true if deleted
     * @throws IOException if deletion fails
     */
    public boolean deleteBackup(String backupName) throws IOException {
        validateModelName(backupName);

        lock.writeLock().lock();
        try {
            Path backupFile = backupsPath.resolve(backupName + ModelSerializer.MODEL_EXTENSION);
            Path metadataFile = backupsPath.resolve(backupName + ModelSerializer.METADATA_EXTENSION);

            boolean deleted = Files.deleteIfExists(backupFile);
            Files.deleteIfExists(metadataFile);

            return deleted;

        } finally {
            lock.writeLock().unlock();
        }
    }

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    private Path getModelPath(String modelName) {
        return modelsPath.resolve(modelName + ModelSerializer.MODEL_EXTENSION);
    }

    private Path getMetadataPath(String modelName) {
        return modelsPath.resolve(modelName + ModelSerializer.METADATA_EXTENSION);
    }

    private String extractModelName(Path modelFile) {
        String filename = modelFile.getFileName().toString();
        return filename.substring(0, filename.length() - ModelSerializer.MODEL_EXTENSION.length());
    }

    private void validateModelName(String modelName) {
        if (modelName == null || modelName.trim().isEmpty()) {
            throw new IllegalArgumentException("Model name cannot be null or empty");
        }
        if (!modelName.matches("^[a-zA-Z0-9][a-zA-Z0-9_-]*$")) {
            throw new IllegalArgumentException(
                    "Invalid model name. Use only alphanumeric characters, hyphens, and underscores. " +
                            "Must start with alphanumeric character.");
        }
        if (modelName.length() > 128) {
            throw new IllegalArgumentException("Model name too long (max 128 characters)");
        }
    }

    private ModelMetadataJson createMetadata(TrainedModel model, String modelName, Path modelFile) throws IOException {
        ModelMetadataJson metadata = new ModelMetadataJson();
        metadata.modelName = modelName;
        metadata.algorithmName = model.getAlgorithmName();
        metadata.algorithmVersion = model.getAlgorithmVersion();
        metadata.modelId = model.getModelId();
        metadata.formatVersion = model.getFormatVersion();
        metadata.trainingTimestamp = model.getTrainingTimestamp();
        metadata.savedTimestamp = LocalDateTime.now();
        metadata.identityCount = model.getIdentityCount();
        metadata.compressed = compressionEnabled;
        metadata.fileSizeBytes = Files.size(modelFile);
        metadata.checksum = ModelSerializer.computeSHA256(modelFile);
        return metadata;
    }

    private ModelInfo buildModelInfo(String modelName, Path modelFile) throws IOException {
        Path metadataFile = getMetadataPath(modelName);

        ModelInfo.Builder builder = ModelInfo.builder()
                .modelName(modelName)
                .filePath(modelFile)
                .fileSizeBytes(Files.size(modelFile));

        // Try to read from JSON metadata first (faster)
        if (Files.exists(metadataFile)) {
            try {
                ModelMetadataJson metadata = objectMapper.readValue(metadataFile.toFile(), ModelMetadataJson.class);
                builder.algorithmName(metadata.algorithmName)
                        .algorithmVersion(metadata.algorithmVersion)
                        .modelId(metadata.modelId)
                        .formatVersion(metadata.formatVersion)
                        .trainingTimestamp(metadata.trainingTimestamp)
                        .savedTimestamp(metadata.savedTimestamp)
                        .identityCount(metadata.identityCount)
                        .compressed(metadata.compressed)
                        .checksum(metadata.checksum);
            } catch (Exception e) {
                logger.warn("Failed to read metadata JSON, falling back to header info", e);
                populateFromHeader(builder, modelFile);
            }
        } else {
            populateFromHeader(builder, modelFile);
        }

        return builder.build();
    }

    private void populateFromHeader(ModelInfo.Builder builder, Path modelFile) throws IOException {
        ModelSerializer.HeaderInfo header = ModelSerializer.getHeaderInfo(modelFile);
        builder.formatVersion(header.getFormatVersion())
                .compressed(header.isCompressed())
                .savedTimestamp(LocalDateTime.ofInstant(
                        Files.getLastModifiedTime(modelFile).toInstant(),
                        ZoneId.systemDefault()));
    }

    // ========================================================================
    // Inner Classes
    // ========================================================================

    /**
     * JSON structure for model metadata file.
     */
    private static class ModelMetadataJson {
        public String modelName;
        public String algorithmName;
        public int algorithmVersion;
        public String modelId;
        public int formatVersion;
        public LocalDateTime trainingTimestamp;
        public LocalDateTime savedTimestamp;
        public int identityCount;
        public boolean compressed;
        public long fileSizeBytes;
        public String checksum;
    }

    /**
     * Configuration options for FileModelRepository.
     */
    public static class Config {
        boolean compressionEnabled = true;
        boolean cachingEnabled = true;
        boolean validateOnLoad = true;

        /**
         * Enables or disables GZIP compression.
         *
         * @param enabled true to enable compression
         * @return this config for chaining
         */
        public Config compressionEnabled(boolean enabled) {
            this.compressionEnabled = enabled;
            return this;
        }

        /**
         * Enables or disables in-memory caching.
         *
         * @param enabled true to enable caching
         * @return this config for chaining
         */
        public Config cachingEnabled(boolean enabled) {
            this.cachingEnabled = enabled;
            return this;
        }

        /**
         * Enables or disables model validation on load.
         *
         * @param enabled true to validate on load
         * @return this config for chaining
         */
        public Config validateOnLoad(boolean enabled) {
            this.validateOnLoad = enabled;
            return this;
        }
    }
}
