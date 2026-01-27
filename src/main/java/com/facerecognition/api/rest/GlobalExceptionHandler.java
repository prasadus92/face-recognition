package com.facerecognition.api.rest;

import com.facerecognition.api.rest.dto.ErrorResponse;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.validation.FieldError;
import org.springframework.web.HttpMediaTypeNotSupportedException;
import org.springframework.web.HttpRequestMethodNotSupportedException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.MissingServletRequestParameterException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;
import org.springframework.web.method.annotation.MethodArgumentTypeMismatchException;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;

import javax.servlet.http.HttpServletRequest;
import javax.validation.ConstraintViolation;
import javax.validation.ConstraintViolationException;
import java.io.IOException;
import java.util.UUID;

/**
 * Global exception handler for the REST API.
 *
 * <p>This class provides centralized exception handling for all REST controllers,
 * converting exceptions into standardized {@link ErrorResponse} objects.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@RestControllerAdvice
public class GlobalExceptionHandler {

    private static final Logger logger = LoggerFactory.getLogger(GlobalExceptionHandler.class);

    /**
     * Handles identity not found exceptions.
     */
    @ExceptionHandler(FaceRecognitionController.IdentityNotFoundException.class)
    public ResponseEntity<ErrorResponse> handleIdentityNotFound(
            FaceRecognitionController.IdentityNotFoundException ex,
            HttpServletRequest request) {

        logger.warn("Identity not found: {}", ex.getIdentityId());

        ErrorResponse error = ErrorResponse.notFound(
                ErrorResponse.ErrorCodes.IDENTITY_NOT_FOUND,
                ex.getMessage(),
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.NOT_FOUND);
    }

    /**
     * Handles training in progress exceptions.
     */
    @ExceptionHandler(TrainingController.TrainingInProgressException.class)
    public ResponseEntity<ErrorResponse> handleTrainingInProgress(
            TrainingController.TrainingInProgressException ex,
            HttpServletRequest request) {

        logger.warn("Training already in progress");

        ErrorResponse error = ErrorResponse.of(
                HttpStatus.CONFLICT.value(),
                "Conflict",
                "TRAINING_IN_PROGRESS",
                ex.getMessage(),
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.CONFLICT);
    }

    /**
     * Handles no training samples exception.
     */
    @ExceptionHandler(TrainingController.NoTrainingSamplesException.class)
    public ResponseEntity<ErrorResponse> handleNoTrainingSamples(
            TrainingController.NoTrainingSamplesException ex,
            HttpServletRequest request) {

        logger.warn("No training samples available");

        ErrorResponse error = ErrorResponse.badRequest(
                ErrorResponse.ErrorCodes.NO_IDENTITIES,
                ex.getMessage(),
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles training failed exception.
     */
    @ExceptionHandler(TrainingController.TrainingFailedException.class)
    public ResponseEntity<ErrorResponse> handleTrainingFailed(
            TrainingController.TrainingFailedException ex,
            HttpServletRequest request) {

        logger.error("Training failed", ex);

        ErrorResponse error = ErrorResponse.internalError(
                ex.getMessage(),
                request.getRequestURI()
        );
        error.setCode(ErrorResponse.ErrorCodes.TRAINING_FAILED);
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.INTERNAL_SERVER_ERROR);
    }

    /**
     * Handles model not trained exception.
     */
    @ExceptionHandler(TrainingController.ModelNotTrainedException.class)
    public ResponseEntity<ErrorResponse> handleModelNotTrained(
            TrainingController.ModelNotTrainedException ex,
            HttpServletRequest request) {

        logger.warn("Model not trained");

        ErrorResponse error = ErrorResponse.badRequest(
                ErrorResponse.ErrorCodes.MODEL_NOT_TRAINED,
                ex.getMessage(),
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles model export exception.
     */
    @ExceptionHandler(TrainingController.ModelExportException.class)
    public ResponseEntity<ErrorResponse> handleModelExport(
            TrainingController.ModelExportException ex,
            HttpServletRequest request) {

        logger.error("Model export failed", ex);

        ErrorResponse error = ErrorResponse.internalError(
                ex.getMessage(),
                request.getRequestURI()
        );
        error.setCode(ErrorResponse.ErrorCodes.EXPORT_FAILED);
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.INTERNAL_SERVER_ERROR);
    }

    /**
     * Handles model import exception.
     */
    @ExceptionHandler(TrainingController.ModelImportException.class)
    public ResponseEntity<ErrorResponse> handleModelImport(
            TrainingController.ModelImportException ex,
            HttpServletRequest request) {

        logger.error("Model import failed", ex);

        ErrorResponse error = ErrorResponse.badRequest(
                ErrorResponse.ErrorCodes.IMPORT_FAILED,
                ex.getMessage(),
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles illegal argument exceptions (validation errors).
     */
    @ExceptionHandler(IllegalArgumentException.class)
    public ResponseEntity<ErrorResponse> handleIllegalArgument(
            IllegalArgumentException ex,
            HttpServletRequest request) {

        logger.warn("Invalid argument: {}", ex.getMessage());

        String code = determineErrorCode(ex.getMessage());

        ErrorResponse error = ErrorResponse.badRequest(
                code,
                ex.getMessage(),
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles illegal state exceptions.
     */
    @ExceptionHandler(IllegalStateException.class)
    public ResponseEntity<ErrorResponse> handleIllegalState(
            IllegalStateException ex,
            HttpServletRequest request) {

        logger.warn("Invalid state: {}", ex.getMessage());

        ErrorResponse error = ErrorResponse.badRequest(
                ErrorResponse.ErrorCodes.MODEL_NOT_TRAINED,
                ex.getMessage(),
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles validation exceptions from @Valid annotations.
     */
    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<ErrorResponse> handleMethodArgumentNotValid(
            MethodArgumentNotValidException ex,
            HttpServletRequest request) {

        logger.warn("Validation failed: {}", ex.getMessage());

        ErrorResponse error = ErrorResponse.badRequest(
                ErrorResponse.ErrorCodes.VALIDATION_ERROR,
                "Validation failed",
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        for (FieldError fieldError : ex.getBindingResult().getFieldErrors()) {
            error.addDetail(
                    fieldError.getField(),
                    fieldError.getRejectedValue(),
                    fieldError.getDefaultMessage()
            );
        }

        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles constraint violation exceptions.
     */
    @ExceptionHandler(ConstraintViolationException.class)
    public ResponseEntity<ErrorResponse> handleConstraintViolation(
            ConstraintViolationException ex,
            HttpServletRequest request) {

        logger.warn("Constraint violation: {}", ex.getMessage());

        ErrorResponse error = ErrorResponse.badRequest(
                ErrorResponse.ErrorCodes.VALIDATION_ERROR,
                "Validation failed",
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        for (ConstraintViolation<?> violation : ex.getConstraintViolations()) {
            error.addDetail(
                    violation.getPropertyPath().toString(),
                    violation.getInvalidValue(),
                    violation.getMessage()
            );
        }

        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles missing request parameter exceptions.
     */
    @ExceptionHandler(MissingServletRequestParameterException.class)
    public ResponseEntity<ErrorResponse> handleMissingParameter(
            MissingServletRequestParameterException ex,
            HttpServletRequest request) {

        logger.warn("Missing parameter: {}", ex.getParameterName());

        ErrorResponse error = ErrorResponse.badRequest(
                ErrorResponse.ErrorCodes.VALIDATION_ERROR,
                "Required parameter '" + ex.getParameterName() + "' is missing",
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());
        error.addDetail(ex.getParameterName(), null, "Parameter is required");

        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles missing multipart file exceptions.
     */
    @ExceptionHandler(MissingServletRequestPartException.class)
    public ResponseEntity<ErrorResponse> handleMissingPart(
            MissingServletRequestPartException ex,
            HttpServletRequest request) {

        logger.warn("Missing request part: {}", ex.getRequestPartName());

        ErrorResponse error = ErrorResponse.badRequest(
                ErrorResponse.ErrorCodes.VALIDATION_ERROR,
                "Required file '" + ex.getRequestPartName() + "' is missing",
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());
        error.addDetail(ex.getRequestPartName(), null, "File is required");

        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles file upload size exceeded exceptions.
     */
    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<ErrorResponse> handleMaxUploadSize(
            MaxUploadSizeExceededException ex,
            HttpServletRequest request) {

        logger.warn("File upload size exceeded: {}", ex.getMessage());

        ErrorResponse error = ErrorResponse.badRequest(
                ErrorResponse.ErrorCodes.IMAGE_TOO_LARGE,
                "File size exceeds maximum allowed size",
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles type mismatch exceptions.
     */
    @ExceptionHandler(MethodArgumentTypeMismatchException.class)
    public ResponseEntity<ErrorResponse> handleTypeMismatch(
            MethodArgumentTypeMismatchException ex,
            HttpServletRequest request) {

        logger.warn("Type mismatch: {} = {}", ex.getName(), ex.getValue());

        ErrorResponse error = ErrorResponse.badRequest(
                ErrorResponse.ErrorCodes.VALIDATION_ERROR,
                "Invalid value for parameter '" + ex.getName() + "'",
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());
        error.addDetail(ex.getName(), ex.getValue(), "Invalid type");

        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles unsupported media type exceptions.
     */
    @ExceptionHandler(HttpMediaTypeNotSupportedException.class)
    public ResponseEntity<ErrorResponse> handleUnsupportedMediaType(
            HttpMediaTypeNotSupportedException ex,
            HttpServletRequest request) {

        logger.warn("Unsupported media type: {}", ex.getContentType());

        ErrorResponse error = ErrorResponse.of(
                HttpStatus.UNSUPPORTED_MEDIA_TYPE.value(),
                "Unsupported Media Type",
                ErrorResponse.ErrorCodes.INVALID_IMAGE_FORMAT,
                "Content type '" + ex.getContentType() + "' is not supported",
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.UNSUPPORTED_MEDIA_TYPE);
    }

    /**
     * Handles method not allowed exceptions.
     */
    @ExceptionHandler(HttpRequestMethodNotSupportedException.class)
    public ResponseEntity<ErrorResponse> handleMethodNotAllowed(
            HttpRequestMethodNotSupportedException ex,
            HttpServletRequest request) {

        logger.warn("Method not allowed: {} on {}", ex.getMethod(), request.getRequestURI());

        ErrorResponse error = ErrorResponse.of(
                HttpStatus.METHOD_NOT_ALLOWED.value(),
                "Method Not Allowed",
                "METHOD_NOT_ALLOWED",
                "Method '" + ex.getMethod() + "' is not supported for this endpoint",
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.METHOD_NOT_ALLOWED);
    }

    /**
     * Handles message not readable exceptions.
     */
    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<ErrorResponse> handleMessageNotReadable(
            HttpMessageNotReadableException ex,
            HttpServletRequest request) {

        logger.warn("Message not readable: {}", ex.getMessage());

        ErrorResponse error = ErrorResponse.badRequest(
                ErrorResponse.ErrorCodes.VALIDATION_ERROR,
                "Malformed request body",
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.BAD_REQUEST);
    }

    /**
     * Handles IO exceptions.
     */
    @ExceptionHandler(IOException.class)
    public ResponseEntity<ErrorResponse> handleIOException(
            IOException ex,
            HttpServletRequest request) {

        logger.error("IO error: {}", ex.getMessage(), ex);

        ErrorResponse error = ErrorResponse.internalError(
                "Failed to process file: " + ex.getMessage(),
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.INTERNAL_SERVER_ERROR);
    }

    /**
     * Handles all other exceptions.
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ErrorResponse> handleGenericException(
            Exception ex,
            HttpServletRequest request) {

        logger.error("Unexpected error: {}", ex.getMessage(), ex);

        ErrorResponse error = ErrorResponse.internalError(
                "An unexpected error occurred",
                request.getRequestURI()
        );
        error.setTraceId(generateTraceId());

        return new ResponseEntity<>(error, HttpStatus.INTERNAL_SERVER_ERROR);
    }

    /**
     * Generates a unique trace ID for error tracking.
     */
    private String generateTraceId() {
        return UUID.randomUUID().toString();
    }

    /**
     * Determines the error code based on the error message.
     */
    private String determineErrorCode(String message) {
        if (message == null) {
            return ErrorResponse.ErrorCodes.VALIDATION_ERROR;
        }

        String lowerMessage = message.toLowerCase();

        if (lowerMessage.contains("no face") || lowerMessage.contains("face not")) {
            return ErrorResponse.ErrorCodes.FACE_NOT_DETECTED;
        }
        if (lowerMessage.contains("multiple face")) {
            return ErrorResponse.ErrorCodes.MULTIPLE_FACES;
        }
        if (lowerMessage.contains("quality")) {
            return ErrorResponse.ErrorCodes.POOR_QUALITY;
        }
        if (lowerMessage.contains("too small") || lowerMessage.contains("dimension")) {
            return ErrorResponse.ErrorCodes.IMAGE_TOO_SMALL;
        }
        if (lowerMessage.contains("too large")) {
            return ErrorResponse.ErrorCodes.IMAGE_TOO_LARGE;
        }
        if (lowerMessage.contains("image") || lowerMessage.contains("format")) {
            return ErrorResponse.ErrorCodes.INVALID_IMAGE_FORMAT;
        }

        return ErrorResponse.ErrorCodes.VALIDATION_ERROR;
    }
}
