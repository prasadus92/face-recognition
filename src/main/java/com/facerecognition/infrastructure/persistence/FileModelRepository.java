package com.facerecognition.infrastructure.persistence;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Pattern;

/** Stores each model as {@code <name>.frm} in one directory, written atomically. */
public class FileModelRepository implements ModelRepository {

    static final String EXTENSION = ".frm";

    private static final Pattern VALID_NAME = Pattern.compile("[A-Za-z0-9_-][A-Za-z0-9._-]{0,63}");

    private final Path directory;

    /**
     * @param directory where model files live; created if missing
     * @throws IOException if the directory cannot be created
     */
    public FileModelRepository(Path directory) throws IOException {
        this.directory = Files.createDirectories(Objects.requireNonNull(directory, "directory"));
    }

    @Override
    public void save(String name, byte[] model) throws IOException {
        Path target = pathFor(name);
        Path temp = Files.createTempFile(directory, name, ".tmp");
        try {
            Files.write(temp, model);
            Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    @Override
    public Optional<byte[]> load(String name) throws IOException {
        try {
            return Optional.of(Files.readAllBytes(pathFor(name)));
        } catch (NoSuchFileException e) {
            return Optional.empty();
        }
    }

    private Path pathFor(String name) {
        if (name == null || !VALID_NAME.matcher(name).matches()) {
            throw new IllegalArgumentException("Invalid model name: " + name);
        }
        return directory.resolve(name + EXTENSION);
    }
}
