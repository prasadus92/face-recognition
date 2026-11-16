package com.facerecognition.domain.model;

import java.awt.image.BufferedImage;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Reads binary (P5) and plain (P2) PGM images, the format of the AT&amp;T/ORL face database,
 * which {@code javax.imageio} does not support. Samples above 8 bits are scaled to 0-255.
 */
final class PgmReader {

    private static final int MAX_DIMENSION = 16_384;

    private PgmReader() {
    }

    static BufferedImage read(InputStream input) throws IOException {
        InputStream in = new BufferedInputStream(input);
        String magic = token(in);
        boolean binary = switch (magic) {
            case "P5" -> true;
            case "P2" -> false;
            default -> throw new IOException("Not a PGM file (magic " + magic + ")");
        };
        int width = number(in, "width", 1, MAX_DIMENSION);
        int height = number(in, "height", 1, MAX_DIMENSION);
        int maxValue = number(in, "maximum value", 1, 65_535);

        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int sample = binary ? binarySample(in, maxValue) : number(in, "pixel", 0, maxValue);
                int v = sample * 255 / maxValue;
                image.setRGB(x, y, (v << 16) | (v << 8) | v);
            }
        }
        return image;
    }

    private static int binarySample(InputStream in, int maxValue) throws IOException {
        int high = in.read();
        if (maxValue < 256) {
            return requireByte(high);
        }
        return (requireByte(high) << 8) | requireByte(in.read());
    }

    private static int requireByte(int b) throws IOException {
        if (b < 0) {
            throw new IOException("PGM pixel data is truncated");
        }
        return b;
    }

    private static int number(InputStream in, String field, int min, int max) throws IOException {
        String token = token(in);
        try {
            int value = Integer.parseInt(token);
            if (value < min || value > max) {
                throw new IOException("PGM " + field + " out of range: " + value);
            }
            return value;
        } catch (NumberFormatException e) {
            throw new IOException("PGM " + field + " is not a number: " + token, e);
        }
    }

    /** Next whitespace-delimited header token, skipping {@code #} comments; consumes one trailing whitespace byte. */
    private static String token(InputStream in) throws IOException {
        StringBuilder token = new StringBuilder();
        int c;
        while ((c = in.read()) != -1) {
            if (c == '#' && token.isEmpty()) {
                while (c != -1 && c != '\n' && c != '\r') {
                    c = in.read();
                }
            } else if (Character.isWhitespace(c)) {
                if (!token.isEmpty()) {
                    return token.toString();
                }
            } else {
                token.append((char) c);
            }
        }
        if (token.isEmpty()) {
            throw new IOException("PGM header is truncated");
        }
        return token.toString();
    }
}
