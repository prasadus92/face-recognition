package com.facerecognition;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

import io.swagger.v3.oas.annotations.OpenAPIDefinition;
import io.swagger.v3.oas.annotations.info.Contact;
import io.swagger.v3.oas.annotations.info.Info;
import io.swagger.v3.oas.annotations.info.License;
import io.swagger.v3.oas.annotations.servers.Server;
import io.swagger.v3.oas.annotations.tags.Tag;

/**
 * Spring Boot entry point for the face-recognition service.
 *
 * <p>All bean wiring lives in
 * {@link com.facerecognition.config.FaceRecognitionAutoConfiguration}; the
 * auto-load-on-startup behaviour is handled by
 * {@link com.facerecognition.config.ModelAutoLoadListener}. Keeping this class
 * dependency-free makes it safe to use in {@code @WebMvcTest} slices.</p>
 */
@SpringBootApplication
@OpenAPIDefinition(
        info = @Info(
                title = "Face Recognition API",
                version = "2.1.0",
                description = "Classical face-recognition library for the JVM — Eigenfaces, "
                        + "Fisherfaces, LBPH — exposed as a Spring Boot REST API.",
                contact = @Contact(
                        name = "Prasad Subrahmanya",
                        email = "prasadus92@gmail.com",
                        url = "https://github.com/prasadus92/face-recognition"
                ),
                license = @License(
                        name = "Apache License, Version 2.0",
                        url = "https://www.apache.org/licenses/LICENSE-2.0.txt"
                )
        ),
        servers = {@Server(url = "http://localhost:8080", description = "Local")},
        tags = {
                @Tag(name = "Face Recognition", description = "Enrolment + identification"),
                @Tag(name = "Model Training", description = "Training + export/import")
        }
)
public class FaceRecognitionApplication {

    public static void main(String[] args) {
        SpringApplication.run(FaceRecognitionApplication.class, args);
    }
}
