package com.facerecognition.infrastructure.persistence;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.DisabledOnOs;
import org.junit.jupiter.api.condition.OS;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;

import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("AtomicFiles")
@DisabledOnOs(OS.WINDOWS)
class AtomicFilesTest {

    @TempDir
    Path dir;

    @Test
    @DisplayName("creates parent folders and makes a new file world-readable")
    void newFile() throws Exception {
        Path target = dir.resolve("models/a.frm");

        AtomicFiles.write(target, new byte[]{1});

        assertThat(Files.readAllBytes(target)).containsExactly(1);
        assertThat(PosixFilePermissions.toString(Files.getPosixFilePermissions(target))).isEqualTo("rw-r--r--");
    }

    @Test
    @DisplayName("keeps the permissions of the file it replaces")
    void keepsPermissions() throws Exception {
        Path target = dir.resolve("a.frm");
        Files.write(target, new byte[]{1});
        Files.setPosixFilePermissions(target, PosixFilePermissions.fromString("rw-rw----"));

        AtomicFiles.write(target, new byte[]{2});

        assertThat(Files.readAllBytes(target)).containsExactly(2);
        assertThat(PosixFilePermissions.toString(Files.getPosixFilePermissions(target))).isEqualTo("rw-rw----");
    }
}
