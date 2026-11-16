package com.facerecognition.api.cli.commands;

import com.facerecognition.api.cli.HelpOption;
import com.facerecognition.benchmark.BenchmarkRunner;
import com.facerecognition.benchmark.BenchmarkRunner.BenchmarkType;
import com.facerecognition.benchmark.BenchmarkRunner.RunnerConfig;
import com.facerecognition.config.FaceRecognitionProperties;
import com.facerecognition.config.FaceRecognitionProperties.ExtractorType;
import com.facerecognition.config.FaceRecognitionServiceFactory;
import com.facerecognition.domain.service.FaceClassifier;
import com.facerecognition.domain.service.FaceClassifier.DistanceMetric;
import com.facerecognition.domain.service.FeatureExtractor;

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
import java.util.function.Supplier;

/** Command-line front end for {@link BenchmarkRunner}. */
@Command(name = "benchmark", description = "Measure accuracy and speed of the algorithms on a labeled dataset.")
public final class BenchmarkCommand implements Callable<Integer> {

    @Spec
    private CommandSpec spec;

    @Mixin
    private HelpOption help;

    @Parameters(paramLabel = "DATASET", description = "Dataset folder.")
    private Path dataset;

    /** Dataset layouts {@link com.facerecognition.benchmark.DatasetLoader} understands. */
    enum Format { ORL, YALE, LFW, FLAT, CUSTOM }

    @Option(names = "--format", defaultValue = "CUSTOM",
            description = "ORL, YALE, LFW, FLAT or CUSTOM (DATASET/<person>/<image>) (default: ${DEFAULT-VALUE}).")
    private Format format;

    @Option(names = {"-a", "--algorithm"}, split = ",", defaultValue = "EIGENFACES,FISHERFACES,LBPH",
            description = "Algorithms to compare (default: ${DEFAULT-VALUE}).")
    private List<ExtractorType> algorithms;

    @Option(names = "--type", defaultValue = "ALL",
            description = "ACCURACY, CROSS_VALIDATION, PERFORMANCE or ALL (default: ${DEFAULT-VALUE}).")
    private BenchmarkType type;

    @Option(names = {"-k", "--folds"}, defaultValue = "10",
            description = "Stratified cross-validation folds (default: ${DEFAULT-VALUE}).")
    private int folds;

    @Option(names = "--train-ratio", defaultValue = "0.7",
            description = "Training share for the single-split accuracy run (default: ${DEFAULT-VALUE}).")
    private double trainRatio;

    @Option(names = {"-t", "--threshold"}, defaultValue = "0",
            description = "Minimum match confidence; 0 measures closed-set rank-1 accuracy (default: ${DEFAULT-VALUE}).")
    private double threshold;

    @Option(names = {"-m", "--metric"}, defaultValue = "EUCLIDEAN",
            description = "EUCLIDEAN, COSINE, MANHATTAN or CHI_SQUARE nearest-neighbor distance (default: ${DEFAULT-VALUE}).")
    private DistanceMetric metric;

    @Option(names = {"-c", "--components"}, defaultValue = "10",
            description = "Eigenfaces or Fisherfaces components (default: ${DEFAULT-VALUE}).")
    private int components;

    @Option(names = "--seed", defaultValue = "42", description = "Random seed for splits (default: ${DEFAULT-VALUE}).")
    private long seed;

    @Option(names = {"-o", "--output"}, defaultValue = "benchmark-results",
            description = "Folder for the HTML, JSON, Markdown and CSV reports (default: ${DEFAULT-VALUE}).")
    private Path output;

    @Override
    public Integer call() throws IOException {
        RunnerConfig config = new RunnerConfig()
                .setCrossValFolds(folds)
                .setTrainRatio(trainRatio)
                .setRecognitionThreshold(threshold)
                .setNumComponents(components)
                .setRandomSeed(seed);
        BenchmarkRunner.Builder runner = BenchmarkRunner.builder()
                .datasetPath(dataset)
                .datasetFormat(format.name())
                .outputDirectory(output)
                .benchmarkType(type)
                .classifierSupplier(() -> classifier(config))
                .config(config);
        for (ExtractorType algorithm : algorithms) {
            Supplier<FeatureExtractor> extractor = () -> extractor(algorithm, config);
            runner.addAlgorithm(extractor.get().getAlgorithmName(), extractor);
        }

        BenchmarkRunner.BenchmarkSuite suite = runner.build().run();
        spec.commandLine().getOut().printf("Reports written to %s%n", suite.getOutputDirectory());
        return 0;
    }

    FaceClassifier classifier(RunnerConfig config) {
        return FaceRecognitionServiceFactory.classifier(properties(null, config));
    }

    private FeatureExtractor extractor(ExtractorType algorithm, RunnerConfig config) {
        return FaceRecognitionServiceFactory.extractor(properties(algorithm, config));
    }

    FaceRecognitionProperties properties(ExtractorType algorithm, RunnerConfig config) {
        FaceRecognitionProperties props = new FaceRecognitionProperties();
        if (algorithm != null) {
            props.getExtraction().setAlgorithm(algorithm);
        }
        props.getExtraction().setNumComponents(config.getNumComponents());
        props.getImage().setTargetWidth(config.getTargetWidth());
        props.getImage().setTargetHeight(config.getTargetHeight());
        props.getClassification().setDistanceMetric(metric);
        props.getRecognition().setThreshold(config.getRecognitionThreshold());
        return props;
    }
}
