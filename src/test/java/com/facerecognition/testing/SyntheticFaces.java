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
