package com.facerecognition.api.cli.commands;

import com.facerecognition.api.cli.FaceRecognitionCLI;
import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.service.FaceClassifier;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.infrastructure.classification.KNNClassifier;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.infrastructure.extraction.FisherfacesExtractor;
import com.facerecognition.infrastructure.extraction.LBPHExtractor;

import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.ParentCommand;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * CLI command for training face recognition models.
 *
 * <p>This command trains a new face recognition model from a dataset
 * directory. The dataset should be organized with subdirectories for
 * each identity containing face images.</p>
 *
 * <h3>Dataset Structure:</h3>
 * <pre>
 * dataset/
 *   person1/
 *     img1.jpg
 *     img2.jpg
 *   person2/
 *     img1.jpg
 *     img2.jpg
 * </pre>
 *
 * <h3>Supported Algorithms:</h3>
 * <ul>
 *   <li><b>eigenfaces</b>: PCA-based, fast, good for controlled conditions</li>
 *   <li><b>fisherfaces</b>: LDA-based, better class separation</li>
 *   <li><b>lbph</b>: Texture-based, robust to lighting changes</li>
 * </ul>
 *
 * <h3>Examples:</h3>
 * <pre>
 * # Train with default settings (eigenfaces)
 * face-recognition train -d dataset/ -o model.dat
 *
 * # Train with LBPH algorithm
 * face-recognition train -d dataset/ -a lbph -o model.dat
 *
 * # Train eigenfaces with 20 components
 * face-recognition train -d dataset/ -a eigenfaces --components 20 -o model.dat
 *
 * # Train with validation split
 * face-recognition train -d dataset/ --validation-split 0.2 -o model.dat
 * </pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Command(
    name = "train",
    aliases = {"fit", "build"},
    description = "Train a face recognition model from a dataset.",
    mixinStandardHelpOptions = true,
    sortOptions = false,
    footer = {
        "",
        "Algorithms:",
        "  eigenfaces    PCA-based algorithm, fast training (default)",
        "  fisherfaces   LDA-based, requires multiple samples per identity",
        "  lbph          Local Binary Patterns, robust to lighting changes",
        "",
        "Dataset Format:",
        "  Organize images in subdirectories by identity:",
        "    dataset/person1/img1.jpg, dataset/person2/img1.jpg, etc.",
        "",
        "Examples:",
        "  @|bold face-recognition train -d faces/ -a eigenfaces -o model.dat|@",
        "  @|bold face-recognition train -d lfw/ -a lbph --components 50|@"
    }
)
public class TrainCommand implements Callable<Integer> {

    @ParentCommand
    private FaceRecognitionCLI parent;

    @Parameters(index = "0", arity = "0..1",
            description = "Dataset directory (alternative to -d)")
    private File datasetParam;

    @Option(names = {"-d", "--dataset"}, paramLabel = "DIR",
            description = "Path to the dataset directory")
    private File datasetDir;

    @Option(names = {"-o", "--output"}, paramLabel = "FILE", required = true,
            description = "Output model file path")
    private File outputFile;

    @Option(names = {"-a", "--algorithm"}, paramLabel = "ALG",
            description = "Recognition algorithm: ${COMPLETION-CANDIDATES} (default: ${DEFAULT-VALUE})",
            defaultValue = "eigenfaces")
    private Algorithm algorithm;

    @Option(names = {"--components", "-c"}, paramLabel = "N",
            description = "Number of components/eigenfaces (default: ${DEFAULT-VALUE})",
            defaultValue = "10")
    private int components;

    @Option(names = {"--grid-x"}, paramLabel = "N",
            description = "LBPH grid X divisions (default: ${DEFAULT-VALUE})",
            defaultValue = "8")
    private int gridX;

    @Option(names = {"--grid-y"}, paramLabel = "N",
            description = "LBPH grid Y divisions (default: ${DEFAULT-VALUE})",
            defaultValue = "8")
    private int gridY;

    @Option(names = {"--validation-split"}, paramLabel = "RATIO",
            description = "Fraction of data for validation (0.0-0.5, default: ${DEFAULT-VALUE})",
            defaultValue = "0.0")
    private double validationSplit;

    @Option(names = {"--min-samples"}, paramLabel = "N",
            description = "Minimum samples per identity (default: ${DEFAULT-VALUE})",
            defaultValue = "1")
    private int minSamples;

    @Option(names = {"--max-samples"}, paramLabel = "N",
            description = "Maximum samples per identity (0 = unlimited, default: ${DEFAULT-VALUE})",
            defaultValue = "0")
    private int maxSamples;

    @Option(names = {"--image-width"}, paramLabel = "W",
            description = "Target image width (default: ${DEFAULT-VALUE})",
            defaultValue = "48")
    private int imageWidth;

    @Option(names = {"--image-height"}, paramLabel = "H",
            description = "Target image height (default: ${DEFAULT-VALUE})",
            defaultValue = "64")
    private int imageHeight;

    @Option(names = {"-v", "--verbose"},
            description = "Show detailed training information")
    private boolean verbose;

    @Option(names = {"--extensions"}, paramLabel = "EXT",
            description = "File extensions to process (default: jpg,jpeg,png,bmp)",
            split = ",",
            defaultValue = "jpg,jpeg,png,bmp")
    private Set<String> extensions;

    @Option(names = {"--shuffle"},
            description = "Shuffle training data before training")
    private boolean shuffle;

    @Option(names = {"--seed"}, paramLabel = "N",
            description = "Random seed for shuffling (default: ${DEFAULT-VALUE})",
            defaultValue = "42")
    private long seed;

    @Option(names = {"--normalize"},
            description = "Normalize feature vectors (default: true)",
            defaultValue = "true", negatable = true)
    private boolean normalize;

    /**
     * Supported recognition algorithms.
     */
    public enum Algorithm {
        eigenfaces, lbph, fisherfaces
    }

    @Override
    public Integer call() throws Exception {
        // Determine input source
        File dataset = datasetDir != null ? datasetDir : datasetParam;
        if (dataset == null) {
            System.err.println("Error: No dataset directory specified. Use -d or provide as argument.");
            return FaceRecognitionCLI.EXIT_INVALID_ARGS;
        }

        if (!dataset.exists() || !dataset.isDirectory()) {
            System.err.println("Error: Dataset directory not found: " + dataset);
            return FaceRecognitionCLI.EXIT_FILE_NOT_FOUND;
        }

        // Validate options
        if (validationSplit < 0.0 || validationSplit > 0.5) {
            System.err.println("Error: Validation split must be between 0.0 and 0.5");
            return FaceRecognitionCLI.EXIT_INVALID_ARGS;
        }

        if (components < 1) {
            System.err.println("Error: Components must be at least 1");
            return FaceRecognitionCLI.EXIT_INVALID_ARGS;
        }

        // Collect training data
        if (!isQuiet()) {
            System.out.println("Scanning dataset: " + dataset);
        }

        Map<String, List<File>> identityFiles = collectDataset(dataset);
        if (identityFiles.isEmpty()) {
            System.err.println("Error: No valid identities found in dataset");
            return FaceRecognitionCLI.EXIT_FILE_NOT_FOUND;
        }

        // Filter by min/max samples
        identityFiles = filterBySampleCount(identityFiles);
        if (identityFiles.isEmpty()) {
            System.err.println("Error: No identities meet the minimum sample requirement (" + minSamples + ")");
            return FaceRecognitionCLI.EXIT_INVALID_ARGS;
        }

        int totalFiles = identityFiles.values().stream().mapToInt(List::size).sum();
        int totalIdentities = identityFiles.size();

        if (!isQuiet()) {
            System.out.println("Found " + totalFiles + " images for " + totalIdentities + " identities");
            if (verbose) {
                System.out.println("\nIdentity distribution:");
                identityFiles.entrySet().stream()
                    .sorted((a, b) -> Integer.compare(b.getValue().size(), a.getValue().size()))
                    .limit(10)
                    .forEach(e -> System.out.printf("  %-20s: %d images%n", e.getKey(), e.getValue().size()));
                if (totalIdentities > 10) {
                    System.out.println("  ... and " + (totalIdentities - 10) + " more identities");
                }
            }
            System.out.println();
        }

        // Fisherfaces requires at least 2 classes
        if (algorithm == Algorithm.fisherfaces && totalIdentities < 2) {
            System.err.println("Error: Fisherfaces requires at least 2 identities");
            return FaceRecognitionCLI.EXIT_INVALID_ARGS;
        }

        // Split validation if requested
        Map<String, List<File>> trainFiles = identityFiles;
        Map<String, List<File>> valFiles = null;

        if (validationSplit > 0) {
            DataSplit split = splitData(identityFiles, validationSplit);
            trainFiles = split.train;
            valFiles = split.validation;

            int trainCount = trainFiles.values().stream().mapToInt(List::size).sum();
            int valCount = valFiles.values().stream().mapToInt(List::size).sum();

            if (!isQuiet()) {
                System.out.printf("Training set: %d images, Validation set: %d images%n%n",
                    trainCount, valCount);
            }
        }

        // Create feature extractor
        FeatureExtractor extractor = createExtractor();

        // Load training images
        if (!isQuiet()) {
            System.out.println("Loading training images...");
        }

        List<FaceImage> trainImages = new ArrayList<>();
        List<String> trainLabels = new ArrayList<>();
        int current = 0;
        int totalTrain = trainFiles.values().stream().mapToInt(List::size).sum();
        int loadErrors = 0;

        for (Map.Entry<String, List<File>> entry : trainFiles.entrySet()) {
            String identity = entry.getKey();
            for (File file : entry.getValue()) {
                current++;
                if (!isQuiet()) {
                    FaceRecognitionCLI.printProgressBar(current, totalTrain, 40);
                }

                try {
                    FaceImage image = FaceImage.fromFile(file);
                    trainImages.add(image);
                    trainLabels.add(identity);
                } catch (Exception e) {
                    loadErrors++;
                    if (verbose) {
                        System.err.println("\nWarning: Could not load " + file.getName() + ": " + e.getMessage());
                    }
                }
            }
        }
        System.out.println();

        if (trainImages.isEmpty()) {
            System.err.println("Error: No images could be loaded for training");
            return FaceRecognitionCLI.EXIT_ERROR;
        }

        // Shuffle if requested
        if (shuffle) {
            if (!isQuiet()) {
                System.out.println("Shuffling training data (seed: " + seed + ")...");
            }
            shuffleData(trainImages, trainLabels, seed);
        }

        // Train model
        if (!isQuiet()) {
            System.out.println("Training " + algorithm + " model...");
            System.out.println("  Images: " + trainImages.size());
            System.out.println("  Identities: " + new HashSet<>(trainLabels).size());
            System.out.println("  Components: " + components);
            System.out.println();
        }

        long startTime = System.currentTimeMillis();

        try {
            extractor.train(trainImages, trainLabels);
        } catch (Exception e) {
            System.err.println("Error during training: " + e.getMessage());
            if (parent != null && parent.isDebug()) {
                e.printStackTrace();
            }
            return FaceRecognitionCLI.EXIT_ERROR;
        }

        long trainTime = System.currentTimeMillis() - startTime;

        if (!isQuiet()) {
            System.out.println("Training completed in " + FaceRecognitionCLI.formatDuration(trainTime));
        }

        // Print training metrics
        if (verbose && extractor instanceof EigenfacesExtractor) {
            EigenfacesExtractor eigenfaces = (EigenfacesExtractor) extractor;
            System.out.println("\nEigenfaces Statistics:");
            System.out.printf("  Feature dimension: %d%n", eigenfaces.getFeatureDimension());
            System.out.printf("  Cumulative variance: %.2f%%%n", eigenfaces.getCumulativeVariance() * 100);

            double[] variance = eigenfaces.getExplainedVarianceRatio();
            System.out.println("  Variance per component:");
            for (int i = 0; i < Math.min(5, variance.length); i++) {
                System.out.printf("    Component %d: %.2f%%%n", i + 1, variance[i] * 100);
            }
        }

        // Build and save complete service
        if (!isQuiet()) {
            System.out.println("\nBuilding recognition service...");
        }

        FaceClassifier classifier = new KNNClassifier();
        FaceRecognitionService service = FaceRecognitionService.builder()
            .extractor(extractor)
            .classifier(classifier)
            .build();

        // Enroll training samples
        if (!isQuiet()) {
            System.out.println("Enrolling training samples...");
        }

        for (int i = 0; i < trainImages.size(); i++) {
            service.enroll(trainImages.get(i), trainLabels.get(i));
        }

        // Train the service (this builds the classifier)
        service.train();

        // Validate if requested
        if (valFiles != null && !valFiles.isEmpty()) {
            if (!isQuiet()) {
                System.out.println("\nRunning validation...");
            }
            runValidation(service, valFiles);
        }

        // Save model
        if (!isQuiet()) {
            System.out.println("\nSaving model to: " + outputFile);
        }

        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(new FileOutputStream(outputFile)))) {
            oos.writeObject(service);
        }

        // Final summary
        if (!isQuiet()) {
            System.out.println("\n" + "=".repeat(50));
            System.out.println("TRAINING SUMMARY");
            System.out.println("=".repeat(50));
            System.out.println("Algorithm:        " + algorithm);
            System.out.println("Components:       " + components);
            System.out.println("Training images:  " + trainImages.size());
            System.out.println("Identities:       " + service.getIdentityCount());
            System.out.println("Training time:    " + FaceRecognitionCLI.formatDuration(trainTime));
            System.out.println("Model file:       " + outputFile.getAbsolutePath());
            System.out.printf("Model size:       %.2f KB%n", outputFile.length() / 1024.0);
            if (loadErrors > 0) {
                System.out.println("Load errors:      " + loadErrors);
            }
        }

        return FaceRecognitionCLI.EXIT_SUCCESS;
    }

    private boolean isQuiet() {
        return parent != null && parent.isQuiet();
    }

    private Map<String, List<File>> collectDataset(File dataset) {
        Map<String, List<File>> result = new LinkedHashMap<>();

        File[] subdirs = dataset.listFiles(File::isDirectory);
        if (subdirs == null) {
            return result;
        }

        for (File subdir : subdirs) {
            String identity = subdir.getName();

            try (Stream<Path> stream = Files.walk(subdir.toPath())) {
                List<File> files = stream
                    .map(Path::toFile)
                    .filter(File::isFile)
                    .filter(this::isImageFile)
                    .sorted()
                    .collect(Collectors.toList());

                if (!files.isEmpty()) {
                    result.put(identity, files);
                }
            } catch (IOException e) {
                System.err.println("Warning: Error scanning " + subdir + ": " + e.getMessage());
            }
        }

        return result;
    }

    private Map<String, List<File>> filterBySampleCount(Map<String, List<File>> data) {
        Map<String, List<File>> filtered = new LinkedHashMap<>();

        for (Map.Entry<String, List<File>> entry : data.entrySet()) {
            List<File> files = entry.getValue();

            if (files.size() < minSamples) {
                if (verbose) {
                    System.err.println("Warning: Skipping " + entry.getKey() +
                        " (only " + files.size() + " samples, minimum is " + minSamples + ")");
                }
                continue;
            }

            if (maxSamples > 0 && files.size() > maxSamples) {
                files = new ArrayList<>(files.subList(0, maxSamples));
            }

            filtered.put(entry.getKey(), files);
        }

        return filtered;
    }

    private boolean isImageFile(File file) {
        String name = file.getName().toLowerCase();
        return extensions.stream().anyMatch(ext -> name.endsWith("." + ext.toLowerCase()));
    }

    private FeatureExtractor createExtractor() {
        FeatureExtractor.ExtractorConfig config = new FeatureExtractor.ExtractorConfig()
            .setNumComponents(components)
            .setNormalize(normalize)
            .setImageWidth(imageWidth)
            .setImageHeight(imageHeight);

        switch (algorithm) {
            case lbph:
                return new LBPHExtractor(gridX, gridY, 1, 8);
            case fisherfaces:
                return new FisherfacesExtractor(config);
            case eigenfaces:
            default:
                return new EigenfacesExtractor(config);
        }
    }

    private static class DataSplit {
        Map<String, List<File>> train;
        Map<String, List<File>> validation;
    }

    private DataSplit splitData(Map<String, List<File>> data, double valRatio) {
        DataSplit split = new DataSplit();
        split.train = new LinkedHashMap<>();
        split.validation = new LinkedHashMap<>();

        Random random = new Random(seed);

        for (Map.Entry<String, List<File>> entry : data.entrySet()) {
            List<File> files = new ArrayList<>(entry.getValue());
            Collections.shuffle(files, random);

            int valCount = Math.max(1, (int) (files.size() * valRatio));
            if (files.size() - valCount < 1) {
                // Not enough for split, use all for training
                split.train.put(entry.getKey(), files);
            } else {
                split.validation.put(entry.getKey(), files.subList(0, valCount));
                split.train.put(entry.getKey(), files.subList(valCount, files.size()));
            }
        }

        return split;
    }

    private void shuffleData(List<FaceImage> images, List<String> labels, long seed) {
        Random random = new Random(seed);
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < images.size(); i++) {
            indices.add(i);
        }
        Collections.shuffle(indices, random);

        List<FaceImage> shuffledImages = new ArrayList<>();
        List<String> shuffledLabels = new ArrayList<>();
        for (int idx : indices) {
            shuffledImages.add(images.get(idx));
            shuffledLabels.add(labels.get(idx));
        }

        images.clear();
        images.addAll(shuffledImages);
        labels.clear();
        labels.addAll(shuffledLabels);
    }

    private void runValidation(FaceRecognitionService service, Map<String, List<File>> valFiles) {
        int correct = 0;
        int total = 0;
        int noFace = 0;

        for (Map.Entry<String, List<File>> entry : valFiles.entrySet()) {
            String expectedIdentity = entry.getKey();

            for (File file : entry.getValue()) {
                try {
                    FaceImage image = FaceImage.fromFile(file);
                    var result = service.recognize(image);

                    if (result.isRecognized()) {
                        String recognized = result.getBestMatch().get().getIdentity().getName();
                        if (recognized.equals(expectedIdentity)) {
                            correct++;
                        }
                    } else if (result.getStatus() ==
                            com.facerecognition.domain.model.RecognitionResult.Status.NO_FACE_DETECTED) {
                        noFace++;
                    }
                    total++;
                } catch (Exception e) {
                    // Skip invalid files
                }
            }
        }

        double accuracy = total > 0 ? (double) correct / total * 100 : 0;

        System.out.println("\nValidation Results:");
        System.out.printf("  Accuracy: %.2f%% (%d/%d)%n", accuracy, correct, total);
        if (noFace > 0) {
            System.out.printf("  No face detected: %d%n", noFace);
        }
    }
}
