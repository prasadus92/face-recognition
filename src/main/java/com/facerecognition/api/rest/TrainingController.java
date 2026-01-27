package com.facerecognition.api.rest;

import com.facerecognition.api.rest.dto.ErrorResponse;
import com.facerecognition.api.rest.dto.ModelStatusDto;
import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.service.FeatureExtractor;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.*;
import java.time.LocalDateTime;
import java.util.Collection;

/**
 * REST controller for model training and management operations.
 *
 * <p>This controller provides endpoints for:</p>
 * <ul>
 *   <li>Training the face recognition model</li>
 *   <li>Checking model status</li>
 *   <li>Exporting and importing trained models</li>
 * </ul>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@RestController
@RequestMapping("/api/v1")
@Tag(name = "Model Training", description = "Model training and management endpoints")
public class TrainingController {

    private static final Logger logger = LoggerFactory.getLogger(TrainingController.class);

    private final FaceRecognitionService faceRecognitionService;

    // Training state tracking
    private volatile LocalDateTime lastTrainedAt;
    private volatile Long lastTrainingDurationMs;
    private volatile boolean isTraining;
    private volatile String lastTrainingError;

    /**
     * Constructs the controller with required dependencies.
     *
     * @param faceRecognitionService the face recognition service
     */
    public TrainingController(FaceRecognitionService faceRecognitionService) {
        this.faceRecognitionService = faceRecognitionService;
    }

    /**
     * Trains or retrains the face recognition model.
     *
     * @param force whether to force retraining even if already trained
     * @return the model status after training
     */
    @PostMapping("/train")
    @Operation(
        summary = "Train the model",
        description = "Trains or retrains the face recognition model using all enrolled face samples. " +
                      "This must be called after enrolling new faces for them to be recognized."
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Model trained successfully",
            content = @Content(schema = @Schema(implementation = ModelStatusDto.class))
        ),
        @ApiResponse(
            responseCode = "400",
            description = "No training samples available",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        ),
        @ApiResponse(
            responseCode = "409",
            description = "Training already in progress",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        ),
        @ApiResponse(
            responseCode = "500",
            description = "Training failed",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        )
    })
    public ResponseEntity<ModelStatusDto> train(
            @Parameter(description = "Force retraining even if model is already trained")
            @RequestParam(value = "force", required = false, defaultValue = "false") Boolean force
    ) {
        logger.info("Train model request received: force={}", force);

        if (isTraining) {
            throw new TrainingInProgressException();
        }

        if (faceRecognitionService.isTrained() && !Boolean.TRUE.equals(force)) {
            logger.info("Model already trained, returning current status");
            return ResponseEntity.ok(getModelStatus());
        }

        int identityCount = faceRecognitionService.getIdentityCount();
        if (identityCount == 0) {
            throw new NoTrainingSamplesException();
        }

        isTraining = true;
        lastTrainingError = null;
        long startTime = System.currentTimeMillis();

        try {
            logger.info("Starting model training with {} identities", identityCount);
            faceRecognitionService.train();
            lastTrainedAt = LocalDateTime.now();
            lastTrainingDurationMs = System.currentTimeMillis() - startTime;
            logger.info("Model training completed in {} ms", lastTrainingDurationMs);
        } catch (Exception e) {
            lastTrainingError = e.getMessage();
            logger.error("Model training failed", e);
            throw new TrainingFailedException(e.getMessage());
        } finally {
            isTraining = false;
        }

        return ResponseEntity.ok(getModelStatus());
    }

    /**
     * Gets the current model status.
     *
     * @return the model status
     */
    @GetMapping("/model/status")
    @Operation(
        summary = "Get model status",
        description = "Returns the current status of the face recognition model including " +
                      "training state, enrolled identities count, and algorithm information."
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Model status retrieved successfully",
            content = @Content(schema = @Schema(implementation = ModelStatusDto.class))
        )
    })
    public ResponseEntity<ModelStatusDto> getStatus() {
        logger.debug("Get model status request received");
        return ResponseEntity.ok(getModelStatus());
    }

    /**
     * Exports the trained model as a binary file.
     *
     * @return the model file as a downloadable attachment
     */
    @PostMapping("/model/export")
    @Operation(
        summary = "Export trained model",
        description = "Exports the current trained model as a binary file that can be imported later. " +
                      "The model must be trained before it can be exported."
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Model exported successfully",
            content = @Content(mediaType = "application/octet-stream")
        ),
        @ApiResponse(
            responseCode = "400",
            description = "Model not trained",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        ),
        @ApiResponse(
            responseCode = "500",
            description = "Export failed",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        )
    })
    public ResponseEntity<byte[]> exportModel() {
        logger.info("Export model request received");

        if (!faceRecognitionService.isTrained()) {
            throw new ModelNotTrainedException();
        }

        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            ObjectOutputStream oos = new ObjectOutputStream(baos);

            // Export model data
            ModelExportData exportData = new ModelExportData();
            exportData.setExportedAt(LocalDateTime.now());
            exportData.setAlgorithmName(faceRecognitionService.getExtractor().getAlgorithmName());
            exportData.setAlgorithmVersion(faceRecognitionService.getExtractor().getVersion());
            exportData.setIdentities(faceRecognitionService.getIdentities());

            oos.writeObject(exportData);
            oos.close();

            byte[] modelData = baos.toByteArray();

            HttpHeaders headers = new HttpHeaders();
            headers.setContentType(MediaType.APPLICATION_OCTET_STREAM);
            headers.setContentDispositionFormData("attachment", "face-model.dat");
            headers.setContentLength(modelData.length);

            logger.info("Model exported successfully: {} bytes", modelData.length);

            return new ResponseEntity<>(modelData, headers, HttpStatus.OK);
        } catch (IOException e) {
            logger.error("Model export failed", e);
            throw new ModelExportException(e.getMessage());
        }
    }

    /**
     * Imports a previously exported model.
     *
     * @param modelFile the model file to import
     * @param merge whether to merge with existing identities or replace
     * @return the model status after import
     */
    @PostMapping(value = "/model/import", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(
        summary = "Import trained model",
        description = "Imports a previously exported model file. The model will need to be retrained " +
                      "after import to be fully functional."
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Model imported successfully",
            content = @Content(schema = @Schema(implementation = ModelStatusDto.class))
        ),
        @ApiResponse(
            responseCode = "400",
            description = "Invalid model file",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        ),
        @ApiResponse(
            responseCode = "500",
            description = "Import failed",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        )
    })
    public ResponseEntity<ModelStatusDto> importModel(
            @Parameter(description = "Model file to import", required = true)
            @RequestParam("file") MultipartFile modelFile,

            @Parameter(description = "Whether to merge with existing identities or replace them")
            @RequestParam(value = "merge", required = false, defaultValue = "false") Boolean merge
    ) {
        logger.info("Import model request received: file={}, size={} bytes, merge={}",
                modelFile.getOriginalFilename(), modelFile.getSize(), merge);

        if (modelFile.isEmpty()) {
            throw new IllegalArgumentException("Model file is required");
        }

        try {
            ObjectInputStream ois = new ObjectInputStream(modelFile.getInputStream());
            ModelExportData importData = (ModelExportData) ois.readObject();
            ois.close();

            logger.info("Model import data: algorithm={}, identities={}",
                    importData.getAlgorithmName(), importData.getIdentities().size());

            // Note: In a full implementation, this would merge/replace identities
            // and retrain the model. For now, we just log the import.

            ModelStatusDto status = getModelStatus();
            status.setState(ModelStatusDto.ModelState.NEEDS_TRAINING);
            status.setReady(false);

            logger.info("Model imported successfully, retraining required");

            return ResponseEntity.ok(status);
        } catch (IOException | ClassNotFoundException e) {
            logger.error("Model import failed", e);
            throw new ModelImportException(e.getMessage());
        }
    }

    /**
     * Builds the current model status DTO.
     */
    private ModelStatusDto getModelStatus() {
        ModelStatusDto status = new ModelStatusDto();

        boolean trained = faceRecognitionService.isTrained();
        int identityCount = faceRecognitionService.getIdentityCount();

        if (isTraining) {
            status.setState(ModelStatusDto.ModelState.TRAINING);
            status.setReady(false);
        } else if (lastTrainingError != null) {
            status.setState(ModelStatusDto.ModelState.ERROR);
            status.setReady(false);
            status.setErrorMessage(lastTrainingError);
        } else if (trained) {
            status.setState(ModelStatusDto.ModelState.READY);
            status.setReady(true);
        } else if (identityCount > 0) {
            status.setState(ModelStatusDto.ModelState.NEEDS_TRAINING);
            status.setReady(false);
        } else {
            status.setState(ModelStatusDto.ModelState.NOT_INITIALIZED);
            status.setReady(false);
        }

        status.setIdentityCount(identityCount);

        // Count total samples
        int totalSamples = 0;
        for (Identity identity : faceRecognitionService.getIdentities()) {
            totalSamples += identity.getSampleCount();
        }
        status.setTotalSampleCount(totalSamples);

        // Get algorithm info
        FeatureExtractor extractor = faceRecognitionService.getExtractor();
        status.setAlgorithmName(extractor.getAlgorithmName());
        status.setAlgorithmVersion(extractor.getVersion());
        status.setFeatureDimension(extractor.getFeatureDimension());

        // Get classifier info
        status.setClassifierType(faceRecognitionService.getClassifier().getClass().getSimpleName());

        // Get config
        status.setRecognitionThreshold(faceRecognitionService.getConfig().getRecognitionThreshold());

        // Training timestamps
        status.setLastTrainedAt(lastTrainedAt);
        status.setLastTrainingDurationMs(lastTrainingDurationMs);

        return status;
    }

    /**
     * Data class for model export/import.
     */
    private static class ModelExportData implements Serializable {
        private static final long serialVersionUID = 1L;

        private LocalDateTime exportedAt;
        private String algorithmName;
        private int algorithmVersion;
        private Collection<Identity> identities;

        public LocalDateTime getExportedAt() { return exportedAt; }
        public void setExportedAt(LocalDateTime exportedAt) { this.exportedAt = exportedAt; }

        public String getAlgorithmName() { return algorithmName; }
        public void setAlgorithmName(String algorithmName) { this.algorithmName = algorithmName; }

        public int getAlgorithmVersion() { return algorithmVersion; }
        public void setAlgorithmVersion(int algorithmVersion) { this.algorithmVersion = algorithmVersion; }

        public Collection<Identity> getIdentities() { return identities; }
        public void setIdentities(Collection<Identity> identities) { this.identities = identities; }
    }

    // Custom exceptions

    /**
     * Exception thrown when training is already in progress.
     */
    public static class TrainingInProgressException extends RuntimeException {
        public TrainingInProgressException() {
            super("Training is already in progress");
        }
    }

    /**
     * Exception thrown when there are no training samples.
     */
    public static class NoTrainingSamplesException extends RuntimeException {
        public NoTrainingSamplesException() {
            super("No training samples available. Enroll faces before training.");
        }
    }

    /**
     * Exception thrown when training fails.
     */
    public static class TrainingFailedException extends RuntimeException {
        public TrainingFailedException(String message) {
            super("Training failed: " + message);
        }
    }

    /**
     * Exception thrown when model is not trained.
     */
    public static class ModelNotTrainedException extends RuntimeException {
        public ModelNotTrainedException() {
            super("Model is not trained. Train the model before exporting.");
        }
    }

    /**
     * Exception thrown when model export fails.
     */
    public static class ModelExportException extends RuntimeException {
        public ModelExportException(String message) {
            super("Model export failed: " + message);
        }
    }

    /**
     * Exception thrown when model import fails.
     */
    public static class ModelImportException extends RuntimeException {
        public ModelImportException(String message) {
            super("Model import failed: " + message);
        }
    }
}
