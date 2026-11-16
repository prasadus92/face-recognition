package com.facerecognition.infrastructure.persistence;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

@DisplayName("FileModelRepository")
class FileModelRepositoryTest {

    @TempDir
    Path dir;

    @Test
    @DisplayName("saves, overwrites and loads by name, leaving no temp files")
    void saveAndLoad() throws Exception {
        FileModelRepository repository = new FileModelRepository(dir.resolve("models"));

        repository.save("default", new byte[]{1, 2});
        repository.save("default", new byte[]{3});

        assertThat(repository.load("default")).hasValueSatisfying(b -> assertThat(b).containsExactly(3));
        try (Stream<Path> files = Files.list(dir.resolve("models"))) {
            assertThat(files).containsExactly(dir.resolve("models").resolve("default.frm"));
        }
    }

    @Test
    @DisplayName("returns empty for a model that was never saved")
    void missingModel() throws Exception {
        assertThat(new FileModelRepository(dir).load("absent")).isEmpty();
    }

    @ParameterizedTest
    @ValueSource(strings = {"", "../escape", "a/b", ".hidden", "name with space"})
    @DisplayName("rejects names that could leave the directory or hide files")
    void rejectsUnsafeNames(String name) throws Exception {
        FileModelRepository repository = new FileModelRepository(dir);

        assertThatIllegalArgumentException().isThrownBy(() -> repository.save(name, new byte[0]));
        assertThatIllegalArgumentException().isThrownBy(() -> repository.load(name));
    }
}
