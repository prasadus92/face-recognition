package com.facerecognition.api.cli.commands;

import com.facerecognition.api.cli.HelpOption;
import com.facerecognition.api.cli.ImageInputs;
import com.facerecognition.config.FaceRecognitionProperties;
import com.facerecognition.config.FaceRecognitionServiceFactory;
import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FaceRegion;
import com.facerecognition.domain.service.FaceDetector;

import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.ParameterException;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.Spec;
import tools.jackson.databind.SerializationFeature;
import tools.jackson.databind.json.JsonMapper;

import javax.imageio.ImageIO;
import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;

/** Finds faces with the same detector the REST API uses. */
@Command(name = "detect", description = "Find faces in images and optionally save a copy with the faces boxed.")
public final class DetectCommand implements Callable<Integer> {

    private static final Color BOX = new Color(0x22, 0xD3, 0xEE);

    @Spec
    private CommandSpec spec;

    @Mixin
    private HelpOption help;

    @Parameters(arity = "1..*", paramLabel = "IMAGE", description = "Image files or folders.")
    private List<Path> images;

    @Option(names = "--min-confidence", defaultValue = "0",
            description = "Drop faces whose support, relative to the best face in the image, is below this "
                    + "(0 to 1, default: ${DEFAULT-VALUE}).")
    private double minConfidence;

    @Option(names = {"-a", "--annotate"}, paramLabel = "OUT",
            description = "Write a copy of the single input image with every face boxed (.png, .jpg or .jpeg).")
    private Path annotate;

    @Option(names = "--json", description = "Print results as JSON.")
    private boolean json;

    /** One detected face. */
    record Face(int x, int y, int width, int height, double confidence) {
    }

    /**
     * All faces found in one image.
     *
     * @param error why the image could not be processed, or null
     */
    record Detection(String image, List<Face> faces, String error) {
    }

    @Override
    public Integer call() throws IOException {
        if (!(minConfidence >= 0 && minConfidence <= 1)) {
            throw new ParameterException(spec.commandLine(), "--min-confidence must be between 0 and 1");
        }
        List<Path> files = ImageInputs.expand(images);
        if (annotate != null && (files.size() != 1 || annotatedFormat(annotate) == null)) {
            throw new ParameterException(spec.commandLine(),
                    "--annotate takes exactly one IMAGE and an output ending in .png, .jpg or .jpeg");
        }
        FaceDetector detector = FaceRecognitionServiceFactory.detector(new FaceRecognitionProperties().getDetection());

        List<Detection> detections = new ArrayList<>();
        for (Path path : files) {
            try {
                FaceImage image = FaceImage.fromFile(path.toFile());
                List<Face> faces = detector.detectFaces(image, minConfidence).stream()
                        .map(r -> new Face(r.getX(), r.getY(), r.getWidth(), r.getHeight(), r.getConfidence()))
                        .toList();
                detections.add(new Detection(path.toString(), faces, null));
                if (annotate != null) {
                    writeAnnotated(image.getImage(), faces, annotate);
                }
            } catch (IOException | RuntimeException e) {
                detections.add(new Detection(path.toString(), List.of(), e.getMessage()));
            }
        }
        print(detections, spec.commandLine().getOut());
        return detections.stream().anyMatch(d -> d.error() != null) ? 1 : 0;
    }

    private void print(List<Detection> detections, PrintWriter out) {
        if (json) {
            out.println(JsonMapper.builder().enable(SerializationFeature.INDENT_OUTPUT).build()
                    .writeValueAsString(detections));
            return;
        }
        for (Detection detection : detections) {
            if (detection.error() != null) {
                out.printf("%s\terror: %s%n", detection.image(), detection.error());
            } else if (detection.faces().isEmpty()) {
                out.printf("%s\tno faces%n", detection.image());
            }
            for (Face face : detection.faces()) {
                out.printf(Locale.ROOT, "%s\t%d,%d,%d,%d\t%.2f%n", detection.image(),
                        face.x(), face.y(), face.width(), face.height(), face.confidence());
            }
        }
    }

    private static void writeAnnotated(BufferedImage source, List<Face> faces, Path target) throws IOException {
        BufferedImage copy = new BufferedImage(source.getWidth(), source.getHeight(), BufferedImage.TYPE_INT_RGB);
        Graphics2D g = copy.createGraphics();
        g.setColor(Color.WHITE);
        g.fillRect(0, 0, copy.getWidth(), copy.getHeight());
        g.drawImage(source, 0, 0, null);
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g.setColor(BOX);
        for (Face face : faces) {
            g.setStroke(new BasicStroke(Math.max(2f, face.width() / 40f)));
            g.drawRoundRect(face.x(), face.y(), face.width(), face.height(), face.width() / 10, face.height() / 10);
        }
        g.dispose();
        if (!ImageIO.write(copy, annotatedFormat(target), target.toFile())) {
            throw new IOException("No image writer for " + target);
        }
    }

    private static String annotatedFormat(Path target) {
        String name = target.getFileName().toString().toLowerCase(Locale.ROOT);
        if (name.endsWith(".png")) {
            return "png";
        }
        return name.endsWith(".jpg") || name.endsWith(".jpeg") ? "jpg" : null;
    }
}
