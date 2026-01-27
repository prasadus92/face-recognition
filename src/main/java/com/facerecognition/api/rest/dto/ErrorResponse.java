package com.facerecognition.api.rest.dto;

import io.swagger.v3.oas.annotations.media.Schema;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.List;

/**
 * Standard error response DTO for the REST API.
 *
 * <p>This class provides a consistent error response format across
 * all API endpoints, including error codes, messages, and details.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Schema(description = "Standard error response format")
public class ErrorResponse {

    @Schema(description = "Timestamp when the error occurred", example = "2024-01-15T10:30:00")
    private LocalDateTime timestamp;

    @Schema(description = "HTTP status code", example = "400")
    private int status;

    @Schema(description = "HTTP status phrase", example = "Bad Request")
    private String error;

    @Schema(description = "Application-specific error code", example = "FACE_NOT_DETECTED")
    private String code;

    @Schema(description = "Human-readable error message", example = "No face was detected in the uploaded image")
    private String message;

    @Schema(description = "Request path that caused the error", example = "/api/v1/recognize")
    private String path;

    @Schema(description = "Unique trace ID for debugging", example = "550e8400-e29b-41d4-a716-446655440000")
    private String traceId;

    @Schema(description = "List of detailed error information")
    private List<ErrorDetail> details;

    /**
     * Detailed error information.
     */
    @Schema(description = "Detailed error information for a specific field or constraint")
    public static class ErrorDetail {

        @Schema(description = "Field that caused the error", example = "image")
        private String field;

        @Schema(description = "Rejected value", example = "null")
        private Object rejectedValue;

        @Schema(description = "Detail message", example = "Image file is required")
        private String message;

        public ErrorDetail() {}

        public ErrorDetail(String field, Object rejectedValue, String message) {
            this.field = field;
            this.rejectedValue = rejectedValue;
            this.message = message;
        }

        public String getField() { return field; }
        public void setField(String field) { this.field = field; }

        public Object getRejectedValue() { return rejectedValue; }
        public void setRejectedValue(Object rejectedValue) { this.rejectedValue = rejectedValue; }

        public String getMessage() { return message; }
        public void setMessage(String message) { this.message = message; }
    }

    /**
     * Common error codes used by the API.
     */
    public static final class ErrorCodes {
        /** No face was detected in the image. */
        public static final String FACE_NOT_DETECTED = "FACE_NOT_DETECTED";
        /** Multiple faces were detected when only one was expected. */
        public static final String MULTIPLE_FACES = "MULTIPLE_FACES";
        /** Image quality is too low for recognition. */
        public static final String POOR_QUALITY = "POOR_QUALITY";
        /** The specified identity was not found. */
        public static final String IDENTITY_NOT_FOUND = "IDENTITY_NOT_FOUND";
        /** The model is not trained. */
        public static final String MODEL_NOT_TRAINED = "MODEL_NOT_TRAINED";
        /** Training failed. */
        public static final String TRAINING_FAILED = "TRAINING_FAILED";
        /** Invalid image format. */
        public static final String INVALID_IMAGE_FORMAT = "INVALID_IMAGE_FORMAT";
        /** Image too small. */
        public static final String IMAGE_TOO_SMALL = "IMAGE_TOO_SMALL";
        /** Image too large. */
        public static final String IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE";
        /** Validation error. */
        public static final String VALIDATION_ERROR = "VALIDATION_ERROR";
        /** Internal server error. */
        public static final String INTERNAL_ERROR = "INTERNAL_ERROR";
        /** No enrolled identities. */
        public static final String NO_IDENTITIES = "NO_IDENTITIES";
        /** Model export failed. */
        public static final String EXPORT_FAILED = "EXPORT_FAILED";
        /** Model import failed. */
        public static final String IMPORT_FAILED = "IMPORT_FAILED";

        private ErrorCodes() {}
    }

    /**
     * Default constructor.
     */
    public ErrorResponse() {
        this.timestamp = LocalDateTime.now();
        this.details = new ArrayList<>();
    }

    /**
     * Creates an error response with the given parameters.
     *
     * @param status HTTP status code
     * @param error HTTP status phrase
     * @param code application error code
     * @param message error message
     * @param path request path
     * @return a new ErrorResponse
     */
    public static ErrorResponse of(int status, String error, String code, String message, String path) {
        ErrorResponse response = new ErrorResponse();
        response.setStatus(status);
        response.setError(error);
        response.setCode(code);
        response.setMessage(message);
        response.setPath(path);
        return response;
    }

    /**
     * Creates a bad request error response.
     *
     * @param code application error code
     * @param message error message
     * @param path request path
     * @return a new ErrorResponse
     */
    public static ErrorResponse badRequest(String code, String message, String path) {
        return of(400, "Bad Request", code, message, path);
    }

    /**
     * Creates a not found error response.
     *
     * @param code application error code
     * @param message error message
     * @param path request path
     * @return a new ErrorResponse
     */
    public static ErrorResponse notFound(String code, String message, String path) {
        return of(404, "Not Found", code, message, path);
    }

    /**
     * Creates an internal server error response.
     *
     * @param message error message
     * @param path request path
     * @return a new ErrorResponse
     */
    public static ErrorResponse internalError(String message, String path) {
        return of(500, "Internal Server Error", ErrorCodes.INTERNAL_ERROR, message, path);
    }

    /**
     * Adds a detail to this error response.
     *
     * @param field the field name
     * @param rejectedValue the rejected value
     * @param message the detail message
     * @return this ErrorResponse for chaining
     */
    public ErrorResponse addDetail(String field, Object rejectedValue, String message) {
        this.details.add(new ErrorDetail(field, rejectedValue, message));
        return this;
    }

    // Getters and Setters

    public LocalDateTime getTimestamp() { return timestamp; }
    public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }

    public int getStatus() { return status; }
    public void setStatus(int status) { this.status = status; }

    public String getError() { return error; }
    public void setError(String error) { this.error = error; }

    public String getCode() { return code; }
    public void setCode(String code) { this.code = code; }

    public String getMessage() { return message; }
    public void setMessage(String message) { this.message = message; }

    public String getPath() { return path; }
    public void setPath(String path) { this.path = path; }

    public String getTraceId() { return traceId; }
    public void setTraceId(String traceId) { this.traceId = traceId; }

    public List<ErrorDetail> getDetails() { return details; }
    public void setDetails(List<ErrorDetail> details) { this.details = details; }
}
