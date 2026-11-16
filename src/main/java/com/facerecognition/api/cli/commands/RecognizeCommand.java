package com.facerecognition.api.cli.commands;

import com.facerecognition.api.cli.HelpOption;
import com.facerecognition.api.cli.ImageInputs;
import com.facerecognition.api.cli.PipelineOptions;
import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.RecognitionResult;

import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.ParameterException;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.Spec;
import tools.jackson.databind.SerializationFeature;
import tools.jackson.databind.json.JsonMapper;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.Callable;

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

    /**
     * One line of output.
     *
     * @param name       the matched person, or null when nobody passed the threshold
     * @param confidence confidence of the nearest person, matched or not
     * @param error      why the image could not be processed, or null
     */
    record Match(String image, String status, String name, double confidence, String error) {
    }

    @Override
    public Integer call() throws IOException {
        if (threshold < 0 || threshold > 1) {
            throw new ParameterException(spec.commandLine(), "--threshold must be between 0 and 1");
        }
        FaceRecognitionService service = pipeline.load(model);
        service.getConfig().setRecognitionThreshold(threshold);

        List<Match> matches = ImageInputs.expand(inputs).stream().map(image -> recognize(service, image)).toList();
        print(matches, spec.commandLine().getOut());
        return matches.stream().anyMatch(m -> m.error() != null) ? 1 : 0;
    }

    private static Match recognize(FaceRecognitionService service, Path image) {
        try {
            RecognitionResult result = service.recognize(FaceImage.fromFile(image.toFile()));
            double nearest = result.getBestMatch().or(() -> result.getAlternatives().stream().findFirst())
                    .map(RecognitionResult.MatchResult::getConfidence)
                    .orElse(0.0);
            return new Match(image.toString(), result.getStatus().name(),
                    result.getBestMatch().map(m -> m.getIdentity().getName()).orElse(null), nearest, null);
        } catch (IOException | RuntimeException e) {
            return new Match(image.toString(), "ERROR", null, 0, e.getMessage());
        }
    }

    private void print(List<Match> matches, PrintWriter out) {
        if (json) {
            out.println(JsonMapper.builder().enable(SerializationFeature.INDENT_OUTPUT).build()
                    .writeValueAsString(matches));
            return;
        }
        for (Match match : matches) {
            String outcome = match.error() != null ? "error: " + match.error()
                    : match.name() != null ? match.name()
                    : match.status().toLowerCase(Locale.ROOT);
            out.printf("%s\t%s\t%.3f%n", match.image(), outcome, match.confidence());
        }
    }
}
