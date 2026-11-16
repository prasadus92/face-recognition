package com.facerecognition.infrastructure.persistence;

import java.io.IOException;
import java.util.Optional;

/** Named storage for encoded model files (see {@link ModelCodec}). */
public interface ModelRepository {

    /**
     * Stores a model, replacing any model with the same name.
     *
     * @param name  model name, letters, digits, {@code .}, {@code _} and {@code -} only
     * @param model the encoded model
     * @throws IOException if the model cannot be written
     */
    void save(String name, byte[] model) throws IOException;

    /**
     * Reads a model.
     *
     * @param name model name
     * @return the encoded model, or empty if none is stored under that name
     * @throws IOException if the model exists but cannot be read
     */
    Optional<byte[]> load(String name) throws IOException;
}
