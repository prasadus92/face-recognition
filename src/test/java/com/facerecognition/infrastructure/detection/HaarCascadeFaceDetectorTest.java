package com.facerecognition.infrastructure.detection;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Optional;

import javax.imageio.ImageIO;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FaceRegion;
import com.facerecognition.infrastructure.detection.haar.HaarCascade;

/**
 * Functional tests for the pure-Java {@link HaarCascadeFaceDetector}. These
 * load the bundled OpenCV frontal-face cascade once (it's a non-trivial XML
 * parse) and reuse the instance across tests to keep the suite fast.
 */
@DisplayName("HaarCascadeFaceDetector")
class HaarCascadeFaceDetectorTest {

    private static HaarCascadeFaceDetector detector;

    @BeforeAll
    static void loadDetector() {
        HaarCascade cascade = HaarCascadeFaceDetector.loadDefaultCascade();
        // Relax the Haar parameters a little so that our single-photo
        // regression tests do not become brittle: minNeighbours=2 matches
        // OpenCV's example code, scaleFactor=1.2 keeps the run fast.
        detector = new HaarCascadeFaceDetector(cascade, 40, 1.2, 2);
    }

    @Test
    @DisplayName("detects the face in the bundled face.png test fixture")
    void detectsBundledFace() throws IOException {
        FaceImage image = loadBundledImage("face.png");
        List<FaceRegion> regions = detector.detectFaces(image);

        assertThat(regions)
                .as("The bundled face.png is a clear frontal portrait; the detector "
                        + "must find at least one face in it or the cascade is not wired up.")
                .isNotEmpty();

        FaceRegion biggest = regions.stream()
                .max((a, b) -> Integer.compare(a.getArea(), b.getArea()))
                .orElseThrow();

        // Sanity bounds on the biggest detection: inside the image, at least
        // as big as our minFaceSize, no more than the whole image.
        assertThat(biggest.getX()).isGreaterThanOrEqualTo(0);
        assertThat(biggest.getY()).isGreaterThanOrEqualTo(0);
        assertThat(biggest.getX() + biggest.getWidth()).isLessThanOrEqualTo(image.getWidth());
        assertThat(biggest.getY() + biggest.getHeight()).isLessThanOrEqualTo(image.getHeight());
        assertThat(biggest.getWidth()).isGreaterThanOrEqualTo(40);
        assertThat(biggest.getHeight()).isGreaterThanOrEqualTo(40);
        assertThat(biggest.getConfidence()).isBetween(0.0, 1.0);
    }

    @Test
    @DisplayName("returns empty list on a uniformly-coloured image")
    void returnsEmptyOnUniformImage() {
        BufferedImage uniform = new BufferedImage(200, 200, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = uniform.createGraphics();
        g.setColor(Color.GRAY);
        g.fillRect(0, 0, 200, 200);
        g.dispose();
        FaceImage image = FaceImage.fromBufferedImage(uniform);

        List<FaceRegion> regions = detector.detectFaces(image);
        assertThat(regions)
                .as("A uniform grey image cannot contain a face — the window-variance "
                        + "early-out must reject every window.")
                .isEmpty();
    }

    @Test
    @DisplayName("returns empty list on an image smaller than the trained window")
    void returnsEmptyOnTooSmallImage() {
        BufferedImage tiny = new BufferedImage(20, 20, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = tiny.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, 20, 20);
        g.dispose();
        FaceImage image = FaceImage.fromBufferedImage(tiny);

        assertThat(detector.detectFaces(image)).isEmpty();
    }

    @Test
    @DisplayName("reports its algorithm name and version metadata")
    void metadata() {
        assertThat(detector.getName()).isEqualTo("HaarCascade");
        assertThat(detector.getVersion()).isNotBlank();
        assertThat(detector.getMinFaceSize()).isGreaterThanOrEqualTo(24);
        assertThat(detector.supportsLandmarks()).isFalse();
        Optional<?> landmarks = detector.detectLandmarks(null, null);
        assertThat(landmarks).isEmpty();
    }

    @Test
    @DisplayName("rejects invalid constructor arguments")
    void rejectsInvalidArguments() {
        HaarCascade cascade = HaarCascadeFaceDetector.loadDefaultCascade();
        assertThatThrownBy(() -> new HaarCascadeFaceDetector(null))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new HaarCascadeFaceDetector(cascade, 24, 1.0, 3))
                .isInstanceOf(IllegalArgumentException.class);
        assertThatThrownBy(() -> new HaarCascadeFaceDetector(cascade, 24, 1.1, 0))
                .isInstanceOf(IllegalArgumentException.class);
    }

    // ---------------------------------------------------------------------

    private static FaceImage loadBundledImage(String resource) throws IOException {
        ClassLoader cl = Thread.currentThread().getContextClassLoader();
        try (InputStream in = cl.getResourceAsStream(resource)) {
            if (in == null) {
                throw new IOException("Bundled test resource not found: " + resource);
            }
            BufferedImage bi = ImageIO.read(in);
            if (bi == null) {
                throw new IOException("Failed to decode image: " + resource);
            }
            return FaceImage.fromBufferedImage(bi);
        }
    }
}
