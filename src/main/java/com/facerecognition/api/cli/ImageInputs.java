package com.facerecognition.api.cli;

import com.facerecognition.domain.model.FaceImage;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;

/** Turns command-line arguments into image files, walking folders. */
public final class ImageInputs {

    private ImageInputs() {
    }

    /**
     * @param inputs files and folders
     * @return the files as given, plus every supported image under each folder, in sorted order
     * @throws IOException if a folder cannot be read
     */
    public static List<Path> expand(List<Path> inputs) throws IOException {
        List<Path> images = new ArrayList<>();
        for (Path input : inputs) {
            if (!Files.isDirectory(input)) {
                images.add(input);
                continue;
            }
            try (Stream<Path> files = Files.walk(input)) {
                files.filter(Files::isRegularFile)
                        .filter(p -> FaceImage.ImageFormat.fromFilename(p.toString()) != FaceImage.ImageFormat.UNKNOWN)
                        .sorted()
                        .forEach(images::add);
            }
        }
        return images;
    }
}
