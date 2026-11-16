package com.facerecognition.testing;

import com.facerecognition.domain.model.FaceImage;

import java.awt.image.BufferedImage;
import java.util.Random;

/** Deterministic grayscale images that are easy to tell apart per identity and noisy per sample. */
public final class SyntheticFaces {

    public static final int SIZE = 48;

    private SyntheticFaces() {
    }

    /**
     * @param identity which identity's pattern to draw
     * @param sample   seed for the per-sample noise
     * @return a {@value #SIZE}x{@value #SIZE} image
     */
    public static FaceImage face(int identity, int sample) {
        return FaceImage.fromBufferedImage(image(identity, sample));
    }

    /** Same pattern as {@link #face(int, int)} with red and green shifted in opposite directions. */
    public static FaceImage colorFace(int identity, int sample) {
        BufferedImage gray = image(identity, sample);
        BufferedImage color = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                int v = gray.getRGB(x, y) & 0xFF;
                int shift = Math.min(Math.min(v, 255 - v), 40);
                color.setRGB(x, y, ((v + shift) << 16) | ((v - shift) << 8) | v);
            }
        }
        return FaceImage.fromBufferedImage(color);
    }

    public static BufferedImage image(int identity, int sample) {
        Random noise = new Random(31L * identity + sample);
        BufferedImage image = new BufferedImage(SIZE, SIZE, BufferedImage.TYPE_INT_RGB);
        double fx = 0.15 + 0.11 * identity;
        double fy = 0.07 + 0.05 * (identity % 3);
        for (int y = 0; y < SIZE; y++) {
            for (int x = 0; x < SIZE; x++) {
                double base = 127 + 100 * Math.sin(fx * x + identity) * Math.cos(fy * y);
                int v = (int) Math.max(0, Math.min(255, base + noise.nextGaussian() * 6));
                image.setRGB(x, y, (v << 16) | (v << 8) | v);
            }
        }
        return image;
    }
}
