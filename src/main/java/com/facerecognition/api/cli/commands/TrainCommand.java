package com.facerecognition.api.cli.commands;

import com.facerecognition.api.cli.HelpOption;
import com.facerecognition.api.cli.PipelineOptions;
import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.benchmark.DatasetLoader;
import com.facerecognition.config.FaceRecognitionProperties;
import com.facerecognition.config.FaceRecognitionProperties.ExtractorType;

import picocli.CommandLine.Command;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.Spec;

import java.io.IOException;
import java.nio.file.Path;
import java.util.concurrent.Callable;

/** Trains a model from a folder with one subfolder of images per person. */
@Command(name = "train", description = "Train a model from DATASET/<person>/<image> and write it to a file.")
public final class TrainCommand implements Callable<Integer> {

    @Spec
    private CommandSpec spec;

    @Mixin
    private HelpOption help;

    @Parameters(paramLabel = "DATASET", description = "Folder with one subfolder of face images per person.")
    private Path dataset;

    @Option(names = {"-o", "--output"}, required = true, paramLabel = "MODEL",
            description = "Model file to write.")
    private Path output;

    @Option(names = {"-a", "--algorithm"}, defaultValue = "EIGENFACES",
            description = "EIGENFACES, FISHERFACES or LBPH (default: ${DEFAULT-VALUE}).")
    private ExtractorType algorithm;

    @Option(names = {"-c", "--components"}, defaultValue = "10",
            description = "Eigenfaces or Fisherfaces components to keep (default: ${DEFAULT-VALUE}).")
    private int components;

    @Mixin
    private PipelineOptions pipeline;

    @Override
    public Integer call() throws IOException {
        long start = System.nanoTime();
        DatasetLoader.LoadedDataset data = new DatasetLoader()
                .setNormalizeSize(false)
                .loadCustomDirectory(dataset.toString());
        if (data.getFaces().isEmpty()) {
            throw new IllegalArgumentException("No images found; expected " + dataset + "/<person>/<image>");
        }

        FaceRecognitionProperties props = new FaceRecognitionProperties();
        props.getExtraction().setAlgorithm(algorithm);
        props.getExtraction().setNumComponents(components);
        FaceRecognitionService service = pipeline.newService(props);
        data.getFaces().forEach(face -> service.enroll(face.getImage(), face.getLabel()));
        service.train();
        PipelineOptions.save(service, output);

        spec.commandLine().getOut().printf("Trained %s on %d images of %d people in %d ms. Model: %s%n",
                service.getExtractor().getAlgorithmName(), data.getFaces().size(), data.getNumClasses(),
                (System.nanoTime() - start) / 1_000_000, output);
        return 0;
    }
}
