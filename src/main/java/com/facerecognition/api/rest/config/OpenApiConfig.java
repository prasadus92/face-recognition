package com.facerecognition.api.rest.config;

import io.swagger.v3.oas.models.Components;
import io.swagger.v3.oas.models.OpenAPI;
import io.swagger.v3.oas.models.info.Contact;
import io.swagger.v3.oas.models.info.Info;
import io.swagger.v3.oas.models.info.License;
import io.swagger.v3.oas.models.security.SecurityRequirement;
import io.swagger.v3.oas.models.security.SecurityScheme;
import io.swagger.v3.oas.models.servers.Server;
import io.swagger.v3.oas.models.tags.Tag;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import java.util.Arrays;
import java.util.List;

/**
 * OpenAPI/Swagger configuration for the Face Recognition REST API.
 *
 * <p>This configuration class sets up the OpenAPI documentation including:</p>
 * <ul>
 *   <li>API metadata (title, version, description)</li>
 *   <li>Contact information</li>
 *   <li>License information</li>
 *   <li>Server URLs</li>
 *   <li>Security schemes</li>
 *   <li>Tags for grouping endpoints</li>
 * </ul>
 *
 * <p>Access the Swagger UI at: {@code /swagger-ui.html}</p>
 * <p>Access the OpenAPI spec at: {@code /v3/api-docs}</p>
 *
 * @author Face Recognition Team
 * @version 2.0
 * @since 2.0
 */
@Configuration
public class OpenApiConfig {

    @Value("${server.port:8080}")
    private int serverPort;

    @Value("${spring.application.name:Face Recognition API}")
    private String applicationName;

    /**
     * Creates the OpenAPI bean with full API documentation.
     *
     * @return the configured OpenAPI instance
     */
    @Bean
    public OpenAPI faceRecognitionOpenAPI() {
        return new OpenAPI()
                .info(apiInfo())
                .servers(serverList())
                .tags(tagList())
                .components(securityComponents())
                .addSecurityItem(securityRequirement());
    }

    /**
     * Creates the API information section.
     */
    private Info apiInfo() {
        return new Info()
                .title("Face Recognition REST API")
                .version("2.0.0")
                .description(buildApiDescription())
                .contact(new Contact()
                        .name("Face Recognition Team")
                        .email("support@facerecognition.com")
                        .url("https://github.com/prasadus92/face-recognition"))
                .license(new License()
                        .name("GNU General Public License v3.0")
                        .url("https://www.gnu.org/licenses/gpl-3.0.html"))
                .termsOfService("https://github.com/prasadus92/face-recognition/blob/main/TERMS.md");
    }

    /**
     * Builds the API description with markdown formatting.
     */
    private String buildApiDescription() {
        return """
            ## Overview

            The Face Recognition API provides a comprehensive set of endpoints for face recognition
            operations including enrollment, recognition, training, and identity management.

            ## Features

            - **Face Recognition**: Identify faces in uploaded images
            - **Face Enrollment**: Register new faces with identities
            - **Identity Management**: Create, update, list, and delete identities
            - **Model Training**: Train and manage the face recognition model
            - **Model Import/Export**: Save and restore trained models

            ## Algorithms

            The API supports multiple face recognition algorithms:
            - **Eigenfaces**: PCA-based feature extraction
            - **Fisherfaces**: LDA-based feature extraction
            - **LBPH**: Local Binary Pattern Histograms

            ## Getting Started

            1. **Enroll faces**: Use `POST /api/v1/enroll` to register face images with identities
            2. **Train the model**: Call `POST /api/v1/train` to train the recognition model
            3. **Recognize faces**: Use `POST /api/v1/recognize` to identify faces in images

            ## Response Codes

            | Code | Description |
            |------|-------------|
            | 200 | Success |
            | 201 | Created |
            | 204 | No Content |
            | 400 | Bad Request - Invalid input |
            | 404 | Not Found - Resource not found |
            | 409 | Conflict - Resource conflict |
            | 500 | Internal Server Error |

            ## Error Codes

            | Code | Description |
            |------|-------------|
            | FACE_NOT_DETECTED | No face found in image |
            | MULTIPLE_FACES | Multiple faces detected |
            | POOR_QUALITY | Image quality too low |
            | IDENTITY_NOT_FOUND | Identity does not exist |
            | MODEL_NOT_TRAINED | Model requires training |
            | VALIDATION_ERROR | Request validation failed |
            """;
    }

    /**
     * Creates the list of server URLs.
     */
    private List<Server> serverList() {
        return Arrays.asList(
                new Server()
                        .url("http://localhost:" + serverPort)
                        .description("Local Development Server"),
                new Server()
                        .url("https://api.facerecognition.com")
                        .description("Production Server")
        );
    }

    /**
     * Creates the list of tags for grouping endpoints.
     */
    private List<Tag> tagList() {
        return Arrays.asList(
                new Tag()
                        .name("Face Recognition")
                        .description("Face recognition and identity management endpoints"),
                new Tag()
                        .name("Model Training")
                        .description("Model training and management endpoints"),
                new Tag()
                        .name("Health")
                        .description("Health check and monitoring endpoints")
        );
    }

    /**
     * Creates security components for API authentication.
     */
    private Components securityComponents() {
        return new Components()
                .addSecuritySchemes("bearerAuth",
                        new SecurityScheme()
                                .type(SecurityScheme.Type.HTTP)
                                .scheme("bearer")
                                .bearerFormat("JWT")
                                .description("JWT Bearer token authentication"))
                .addSecuritySchemes("apiKeyAuth",
                        new SecurityScheme()
                                .type(SecurityScheme.Type.APIKEY)
                                .in(SecurityScheme.In.HEADER)
                                .name("X-API-Key")
                                .description("API Key authentication"));
    }

    /**
     * Creates the security requirement for protected endpoints.
     */
    private SecurityRequirement securityRequirement() {
        return new SecurityRequirement()
                .addList("bearerAuth")
                .addList("apiKeyAuth");
    }
}
