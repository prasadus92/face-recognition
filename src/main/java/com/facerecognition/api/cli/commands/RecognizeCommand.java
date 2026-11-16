package com.facerecognition.api.cli.commands;

import com.facerecognition.api.cli.HelpOption;
import com.facerecognition.api.cli.PipelineOptions;
import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.RecognitionResult;

import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.Spec;
import tools.jackson.databind.SerializationFeature;
import tools.jackson.databind.json.JsonMapper;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.stream.Stream;

/** Identifies the person in each image. */
@Command(name = "recognize", description = "Identify the person in each image, or in every image under a folder.")
public final class RecognizeCommand implements Callable<Integer> {

    @Spec
    private CommandSpec spec;

    @Mixin
    private HelpOption help;

    @Option(names = {"-m", "--model"}, required = true, paramLabel = "MODEL", description = "Trained model file.")
    private Path model;

    @Option(names = {"-t", "--threshold"}, defaultValue = "0.6",
            description = "Minimum confidence for a match, 0 to 1 (default: ${DEFAULT-VALUE}).")
    private double threshold;

    @Option(names = "--json", description = "Print results as JSON.")
    private boolean json;

    @Parameters(arity = "1..*", paramLabel = "IMAGE", description = "Image files or folders.")
    private List<Path> inputs;

    @Mixin
    private PipelineOptions pipeline;

    /** One line of output. {@code name} is null when nobody matched. */
    record Match(String image, String status, String name, double confidence) {
    }

    @Override
    public Integer call() throws IOException {
        FaceRecognitionService service = pipeline.load(model);
        service.getConfig().setRecognitionThreshold(threshold);

        List<Match> matches = new ArrayList<>();
        for (Path image : expand(inputs)) {
            RecognitionResult result = service.recognize(FaceImage.fromFile(image.toFile()));
            matches.add(new Match(image.toString(), result.getStatus().name(),
                    result.getBestMatch().map(m -> m.getIdentity().getName()).orElse(null),
                    result.getConfidence()));
        }
        print(matches, spec.commandLine().getOut());
        return 0;
    }

    private void print(List<Match> matches, PrintWriter out) {
        if (json) {
            out.println(JsonMapper.builder().enable(SerializationFeature.INDENT_OUTPUT).build()
                    .writeValueAsString(matches));
            return;
        }
        for (Match match : matches) {
            out.printf("%s\t%s\t%.3f%n", match.image(),
                    match.name() == null ? match.status().toLowerCase() : match.name(), match.confidence());
        }
    }

    private static List<Path> expand(List<Path> inputs) throws IOException {
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
