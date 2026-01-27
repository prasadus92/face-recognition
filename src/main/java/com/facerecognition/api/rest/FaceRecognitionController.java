package com.facerecognition.api.rest;

import com.facerecognition.api.rest.dto.*;
import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.Parameter;
import io.swagger.v3.oas.annotations.media.Content;
import io.swagger.v3.oas.annotations.media.Schema;
import io.swagger.v3.oas.annotations.responses.ApiResponse;
import io.swagger.v3.oas.annotations.responses.ApiResponses;
import io.swagger.v3.oas.annotations.tags.Tag;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.validation.annotation.Validated;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import javax.imageio.ImageIO;
import javax.validation.Valid;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.Collection;
import java.util.List;
import java.util.stream.Collectors;

/**
 * REST controller for face recognition operations.
 *
 * <p>This controller provides endpoints for:</p>
 * <ul>
 *   <li>Face recognition - Identify a face in an image</li>
 *   <li>Face enrollment - Register a new face with an identity</li>
 *   <li>Identity management - List, view, and delete identities</li>
 * </ul>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@RestController
@RequestMapping("/api/v1")
@Validated
@Tag(name = "Face Recognition", description = "Face recognition and identity management endpoints")
public class FaceRecognitionController {

    private static final Logger logger = LoggerFactory.getLogger(FaceRecognitionController.class);

    private final FaceRecognitionService faceRecognitionService;

    /**
     * Constructs the controller with required dependencies.
     *
     * @param faceRecognitionService the face recognition service
     */
    public FaceRecognitionController(FaceRecognitionService faceRecognitionService) {
        this.faceRecognitionService = faceRecognitionService;
    }

    /**
     * Recognizes a face in the uploaded image.
     *
     * @param image the image file containing a face
     * @param threshold recognition confidence threshold (optional)
     * @param maxAlternatives maximum number of alternative matches (optional)
     * @param includeFeatures whether to include feature vector in response (optional)
     * @return the recognition result
     */
    @PostMapping(value = "/recognize", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(
        summary = "Recognize a face",
        description = "Analyzes the uploaded image to identify the person. Returns the best match " +
                      "along with confidence score and optional alternative matches."
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Recognition completed successfully",
            content = @Content(schema = @Schema(implementation = RecognitionResponse.class))
        ),
        @ApiResponse(
            responseCode = "400",
            description = "Invalid image or no face detected",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        ),
        @ApiResponse(
            responseCode = "500",
            description = "Internal server error",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        )
    })
    public ResponseEntity<RecognitionResponse> recognize(
            @Parameter(description = "Image file containing a face", required = true)
            @RequestParam("image") MultipartFile image,

            @Parameter(description = "Minimum confidence threshold (0.0 to 1.0)")
            @RequestParam(value = "threshold", required = false, defaultValue = "0.6") Double threshold,

            @Parameter(description = "Maximum number of alternative matches to return")
            @RequestParam(value = "maxAlternatives", required = false, defaultValue = "5") Integer maxAlternatives,

            @Parameter(description = "Whether to include feature vector in response")
            @RequestParam(value = "includeFeatures", required = false, defaultValue = "false") Boolean includeFeatures
    ) throws IOException {
        logger.info("Recognition request received: file={}, size={} bytes",
                image.getOriginalFilename(), image.getSize());

        validateImageFile(image);

        BufferedImage bufferedImage = ImageIO.read(image.getInputStream());
        if (bufferedImage == null) {
            throw new IllegalArgumentException("Could not read image file");
        }

        FaceImage faceImage = FaceImage.fromBufferedImage(bufferedImage);
        RecognitionResult result = faceRecognitionService.recognize(faceImage);

        RecognitionResponse response = RecognitionResponse.fromDomain(result);

        if (!Boolean.TRUE.equals(includeFeatures)) {
            response.setFeatures(null);
        }

        logger.info("Recognition completed: status={}, recognized={}",
                result.getStatus(), result.isRecognized());

        return ResponseEntity.ok(response);
    }

    /**
     * Enrolls a new face with an identity.
     *
     * @param image the image file containing a face
     * @param name the identity name
     * @param externalId optional external system ID
     * @param sourceDescription optional description of the image source
     * @return the enrollment result
     */
    @PostMapping(value = "/enroll", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    @Operation(
        summary = "Enroll a new face",
        description = "Registers a face image with an identity. If the identity already exists, " +
                      "the face is added as an additional sample. The model must be retrained " +
                      "after enrollment for the new face to be recognized."
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "201",
            description = "Face enrolled successfully",
            content = @Content(schema = @Schema(implementation = EnrollmentResponse.class))
        ),
        @ApiResponse(
            responseCode = "400",
            description = "Invalid image or enrollment parameters",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        ),
        @ApiResponse(
            responseCode = "500",
            description = "Internal server error",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        )
    })
    public ResponseEntity<EnrollmentResponse> enroll(
            @Parameter(description = "Image file containing a face", required = true)
            @RequestParam("image") MultipartFile image,

            @Parameter(description = "Name of the person being enrolled", required = true)
            @RequestParam("name") String name,

            @Parameter(description = "Optional external system identifier")
            @RequestParam(value = "externalId", required = false) String externalId,

            @Parameter(description = "Optional description of the image source")
            @RequestParam(value = "sourceDescription", required = false) String sourceDescription
    ) throws IOException {
        logger.info("Enrollment request received: name={}, file={}, size={} bytes",
                name, image.getOriginalFilename(), image.getSize());

        validateImageFile(image);

        if (name == null || name.trim().isEmpty()) {
            throw new IllegalArgumentException("Name is required");
        }

        BufferedImage bufferedImage = ImageIO.read(image.getInputStream());
        if (bufferedImage == null) {
            throw new IllegalArgumentException("Could not read image file");
        }

        FaceImage faceImage = FaceImage.fromBufferedImage(bufferedImage);
        double qualityScore = faceImage.getQualityScore();

        Identity identity = faceRecognitionService.enroll(faceImage, name.trim(), externalId);

        // Get the latest sample ID
        String sampleId = identity.getSamples().isEmpty() ? null :
                identity.getSamples().get(identity.getSamples().size() - 1).getSampleId();

        EnrollmentResponse response = EnrollmentResponse.success(identity, sampleId, qualityScore);

        logger.info("Enrollment completed: identityId={}, name={}, sampleCount={}",
                identity.getId(), identity.getName(), identity.getSampleCount());

        return ResponseEntity.status(HttpStatus.CREATED).body(response);
    }

    /**
     * Removes an identity from the system.
     *
     * @param id the identity ID to remove
     * @return empty response on success
     */
    @DeleteMapping("/identities/{id}")
    @Operation(
        summary = "Remove an identity",
        description = "Deletes an identity and all its enrolled face samples from the system. " +
                      "The model should be retrained after deletion."
    )
    @ApiResponses(value = {
        @ApiResponse(responseCode = "204", description = "Identity deleted successfully"),
        @ApiResponse(
            responseCode = "404",
            description = "Identity not found",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        )
    })
    public ResponseEntity<Void> deleteIdentity(
            @Parameter(description = "Identity ID to delete", required = true)
            @PathVariable("id") String id
    ) {
        logger.info("Delete identity request received: id={}", id);

        Identity identity = findIdentityById(id);
        if (identity == null) {
            throw new IdentityNotFoundException(id);
        }

        // Mark as inactive (soft delete)
        identity.setActive(false);

        logger.info("Identity deleted: id={}, name={}", identity.getId(), identity.getName());

        return ResponseEntity.noContent().build();
    }

    /**
     * Lists all enrolled identities.
     *
     * @param activeOnly whether to return only active identities
     * @return list of identity summaries
     */
    @GetMapping("/identities")
    @Operation(
        summary = "List all identities",
        description = "Returns a list of all enrolled identities with summary information."
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "List of identities retrieved successfully",
            content = @Content(schema = @Schema(implementation = IdentityDto.class))
        )
    })
    public ResponseEntity<List<IdentityDto>> listIdentities(
            @Parameter(description = "Filter to return only active identities")
            @RequestParam(value = "activeOnly", required = false, defaultValue = "true") Boolean activeOnly
    ) {
        logger.debug("List identities request received: activeOnly={}", activeOnly);

        Collection<Identity> identities = faceRecognitionService.getIdentities();

        List<IdentityDto> result = identities.stream()
                .filter(identity -> !Boolean.TRUE.equals(activeOnly) || identity.isActive())
                .map(IdentityDto::summary)
                .collect(Collectors.toList());

        logger.debug("Returning {} identities", result.size());

        return ResponseEntity.ok(result);
    }

    /**
     * Gets detailed information about a specific identity.
     *
     * @param id the identity ID
     * @return identity details including all enrolled samples
     */
    @GetMapping("/identities/{id}")
    @Operation(
        summary = "Get identity details",
        description = "Returns detailed information about a specific identity including all enrolled samples."
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Identity details retrieved successfully",
            content = @Content(schema = @Schema(implementation = IdentityDto.class))
        ),
        @ApiResponse(
            responseCode = "404",
            description = "Identity not found",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        )
    })
    public ResponseEntity<IdentityDto> getIdentity(
            @Parameter(description = "Identity ID", required = true)
            @PathVariable("id") String id
    ) {
        logger.debug("Get identity request received: id={}", id);

        Identity identity = findIdentityById(id);
        if (identity == null) {
            throw new IdentityNotFoundException(id);
        }

        IdentityDto response = IdentityDto.fromDomain(identity);

        return ResponseEntity.ok(response);
    }

    /**
     * Updates an identity's information.
     *
     * @param id the identity ID
     * @param name new name (optional)
     * @param externalId new external ID (optional)
     * @param active active status (optional)
     * @return updated identity details
     */
    @PatchMapping("/identities/{id}")
    @Operation(
        summary = "Update identity",
        description = "Updates an identity's name, external ID, or active status."
    )
    @ApiResponses(value = {
        @ApiResponse(
            responseCode = "200",
            description = "Identity updated successfully",
            content = @Content(schema = @Schema(implementation = IdentityDto.class))
        ),
        @ApiResponse(
            responseCode = "404",
            description = "Identity not found",
            content = @Content(schema = @Schema(implementation = ErrorResponse.class))
        )
    })
    public ResponseEntity<IdentityDto> updateIdentity(
            @Parameter(description = "Identity ID", required = true)
            @PathVariable("id") String id,

            @Parameter(description = "New name for the identity")
            @RequestParam(value = "name", required = false) String name,

            @Parameter(description = "New external ID")
            @RequestParam(value = "externalId", required = false) String externalId,

            @Parameter(description = "Active status")
            @RequestParam(value = "active", required = false) Boolean active
    ) {
        logger.info("Update identity request received: id={}", id);

        Identity identity = findIdentityById(id);
        if (identity == null) {
            throw new IdentityNotFoundException(id);
        }

        if (name != null && !name.trim().isEmpty()) {
            identity.setName(name.trim());
        }
        if (externalId != null) {
            identity.setExternalId(externalId);
        }
        if (active != null) {
            identity.setActive(active);
        }

        IdentityDto response = IdentityDto.fromDomain(identity);

        logger.info("Identity updated: id={}, name={}", identity.getId(), identity.getName());

        return ResponseEntity.ok(response);
    }

    /**
     * Finds an identity by ID from the service.
     */
    private Identity findIdentityById(String id) {
        return faceRecognitionService.getIdentities().stream()
                .filter(identity -> identity.getId().equals(id))
                .findFirst()
                .orElse(null);
    }

    /**
     * Validates the uploaded image file.
     */
    private void validateImageFile(MultipartFile file) {
        if (file == null || file.isEmpty()) {
            throw new IllegalArgumentException("Image file is required");
        }

        String contentType = file.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            throw new IllegalArgumentException("File must be an image");
        }

        // Check file size (max 10MB)
        if (file.getSize() > 10 * 1024 * 1024) {
            throw new IllegalArgumentException("Image file is too large (max 10MB)");
        }
    }

    /**
     * Exception thrown when an identity is not found.
     */
    public static class IdentityNotFoundException extends RuntimeException {
        private final String identityId;

        public IdentityNotFoundException(String identityId) {
            super("Identity not found: " + identityId);
            this.identityId = identityId;
        }

        public String getIdentityId() {
            return identityId;
        }
    }
}
