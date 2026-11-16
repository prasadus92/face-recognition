package com.facerecognition.infrastructure.persistence;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

/** Writes files so readers see either the old content or the new content, never a partial file. */
public final class AtomicFiles {

    private AtomicFiles() {
    }

    /**
     * Writes {@code data} to a temp file next to {@code target}, then moves it into place.
     *
     * @param target the file to create or replace; its parent directories are created
     * @param data   the new content
     * @throws IOException if the file cannot be written
     */
    public static void write(Path target, byte[] data) throws IOException {
        Path directory = target.toAbsolutePath().getParent();
        Files.createDirectories(directory);
        Path temp = Files.createTempFile(directory, target.getFileName().toString(), ".tmp");
        try {
            Files.write(temp, data);
            Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } finally {
            Files.deleteIfExists(temp);
        }
    }
}
