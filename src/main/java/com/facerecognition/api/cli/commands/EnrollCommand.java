package com.facerecognition.api.cli.commands;

import com.facerecognition.api.cli.HelpOption;
import com.facerecognition.api.cli.PipelineOptions;
import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.model.FaceImage;

import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.Spec;

import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Callable;

/** Adds images of one person to an existing model and retrains it. */
@Command(name = "enroll", description = "Add face images of one person to a model and retrain it.")
public final class EnrollCommand implements Callable<Integer> {

    @Spec
    private CommandSpec spec;

    @Mixin
    private HelpOption help;

    @Option(names = {"-m", "--model"}, required = true, paramLabel = "MODEL",
            description = "Model file to update in place.")
    private Path model;

    @Option(names = {"-n", "--name"}, required = true, description = "Name of the person in the images.")
    private String name;

    @Parameters(arity = "1..*", paramLabel = "IMAGE", description = "Face images of that person.")
    private List<Path> images;

    @Mixin
    private PipelineOptions pipeline;

    @Override
    public Integer call() throws IOException {
        FaceRecognitionService service = pipeline.load(model);
        for (Path image : images) {
            service.enroll(FaceImage.fromFile(image.toFile()), name);
        }
        service.train();
        PipelineOptions.save(service, model);

        spec.commandLine().getOut().printf("Enrolled %d images of %s. The model now knows %d people.%n",
                images.size(), name, service.getIdentityCount());
        return 0;
    }
}
