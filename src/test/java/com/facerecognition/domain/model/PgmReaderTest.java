package com.facerecognition.domain.model;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@DisplayName("PgmReader")
class PgmReaderTest {

    @Test
    @DisplayName("reads binary P5 with a comment in the header")
    void readsBinary() throws IOException {
        BufferedImage image = read(concat("P5\n# made by a scanner\n2 1\n255\n", new byte[]{0, (byte) 200}));

        assertThat(image.getWidth()).isEqualTo(2);
        assertThat(image.getHeight()).isEqualTo(1);
        assertThat(gray(image, 0, 0)).isZero();
        assertThat(gray(image, 1, 0)).isEqualTo(200);
    }

    @Test
    @DisplayName("reads plain P2 with the header on one line")
    void readsPlain() throws IOException {
        BufferedImage image = read("P2 3 1 15\n0 15\n7".getBytes(StandardCharsets.US_ASCII));

        assertThat(gray(image, 0, 0)).isZero();
        assertThat(gray(image, 1, 0)).isEqualTo(255);
        assertThat(gray(image, 2, 0)).isEqualTo(7 * 255 / 15);
    }

    @Test
    @DisplayName("scales 16-bit samples to 8 bits")
    void scalesSixteenBit() throws IOException {
        BufferedImage image = read(concat("P5 1 1 65535\n", new byte[]{(byte) 0x80, 0x00}));

        assertThat(gray(image, 0, 0)).isEqualTo(0x8000 * 255 / 65535);
    }

    @ParameterizedTest
    @ValueSource(strings = {"P6 1 1 255\n", "P5 0 1 255\n", "P5 1 1 70000\n",
        "P5 1 1 255\n", "P5 x 1 255\n", "P5"})
    @DisplayName("rejects other formats, bad headers and truncated data")
    void rejectsMalformed(String header) {
        assertThatThrownBy(() -> read(header.getBytes(StandardCharsets.US_ASCII)))
                .isInstanceOf(IOException.class);
    }

    @Test
    @DisplayName("rejects images wider than FaceImage supports, even with complete pixel data")
    void rejectsOversizedImages() {
        int width = FaceImage.MAX_DIMENSION + 1;

        assertThatThrownBy(() -> read(concat("P5 " + width + " 1 255\n", new byte[width])))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("out of range");
    }

    @Test
    @DisplayName("rejects binary samples above the declared maximum")
    void rejectsSamplesAboveMax() {
        assertThatThrownBy(() -> read(concat("P5 1 1 100\n", new byte[]{(byte) 200})))
                .isInstanceOf(IOException.class)
                .hasMessageContaining("exceeds the maximum value");
    }

    private static BufferedImage read(byte[] data) throws IOException {
        return PgmReader.read(new ByteArrayInputStream(data));
    }

    private static byte[] concat(String header, byte[] pixels) {
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        out.writeBytes(header.getBytes(StandardCharsets.US_ASCII));
        out.writeBytes(pixels);
        return out.toByteArray();
    }

    private static int gray(BufferedImage image, int x, int y) {
        return image.getRGB(x, y) & 0xFF;
    }
}
