package com.facerecognition.infrastructure.persistence;

import java.io.IOException;

/** Thrown when bytes cannot be decoded into a {@link TrainedModel}. */
public class InvalidModelException extends IOException {

    private static final long serialVersionUID = 1L;

    public InvalidModelException(String message) {
        super(message);
    }

    public InvalidModelException(String message, Throwable cause) {
        super(message, cause);
    }
}
