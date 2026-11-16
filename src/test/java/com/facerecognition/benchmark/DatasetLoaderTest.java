package com.facerecognition.benchmark;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

@DisplayName("DatasetLoader")
class DatasetLoaderTest {

    @TempDir
    Path dir;

    @Test
    @DisplayName("names the line when an LFW pairs file has a non-numeric image index")
    void rejectsMalformedPairs() throws IOException {
        Path pairs = Files.writeString(dir.resolve("pairs.txt"), "10 300\nAaron_Peirsol one 2\n");

        assertThatThrownBy(() -> new DatasetLoader().loadLfwPairs(dir, pairs))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("Malformed line 2");
    }
}
