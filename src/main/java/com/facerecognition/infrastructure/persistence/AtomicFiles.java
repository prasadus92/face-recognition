package com.facerecognition.infrastructure.persistence;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFileAttributeView;
import java.nio.file.attribute.PosixFilePermission;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.Set;

/** Writes files so readers see either the old content or the new content, never a partial file. */
public final class AtomicFiles {

    private static final Set<PosixFilePermission> NEW_FILE_PERMISSIONS = PosixFilePermissions.fromString("rw-r--r--");

    private AtomicFiles() {
    }

    /**
     * Writes {@code data} to a temp file next to {@code target}, then moves it into place. The file
     * keeps the permissions of the one it replaces, or gets {@code rw-r--r--} when new.
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
            copyPermissions(target, temp);
            Files.move(temp, target, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    private static void copyPermissions(Path target, Path temp) throws IOException {
        PosixFileAttributeView view = Files.getFileAttributeView(temp, PosixFileAttributeView.class);
        if (view == null) {
            return;
        }
        view.setPermissions(Files.exists(target, LinkOption.NOFOLLOW_LINKS)
                ? Files.getPosixFilePermissions(target)
                : NEW_FILE_PERMISSIONS);
    }
}
