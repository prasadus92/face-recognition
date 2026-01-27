package com.facerecognition.api.cli.commands;

import com.facerecognition.api.cli.FaceRecognitionCLI;
import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.domain.service.FaceClassifier;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.infrastructure.classification.KNNClassifier;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.infrastructure.extraction.FisherfacesExtractor;
import com.facerecognition.infrastructure.extraction.LBPHExtractor;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;
import picocli.CommandLine.ParentCommand;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * CLI command for running benchmarks on face recognition algorithms.
 *
 * <p>This command evaluates recognition accuracy using k-fold cross-validation
 * and computes comprehensive metrics including accuracy, precision, recall,
 * F1 score, and confusion matrices.</p>
 *
 * <h3>Evaluation Methodology:</h3>
 * <ul>
 *   <li><b>K-Fold Cross-Validation</b>: Dataset split into k folds, each used as test set once</li>
 *   <li><b>Recognition Rate</b>: Percentage of correctly identified faces</li>
 *   <li><b>False Acceptance Rate (FAR)</b>: Impostor accepted as genuine</li>
 *   <li><b>False Rejection Rate (FRR)</b>: Genuine rejected as impostor</li>
 * </ul>
 *
 * <h3>Examples:</h3>
 * <pre>
 * # Run benchmark with default settings
 * face-recognition benchmark -d lfw/ -a eigenfaces
 *
 * # Run 10-fold cross-validation
 * face-recognition benchmark -d dataset/ -a lbph --folds 10
 *
 * # Compare all algorithms
 * face-recognition benchmark -d dataset/ --compare-all
 *
 * # Output results to JSON
 * face-recognition benchmark -d dataset/ --output benchmark.json
 * </pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Command(
    name = "benchmark",
    aliases = {"bench", "evaluate", "test"},
    description = "Run accuracy benchmarks on face recognition algorithms.",
    mixinStandardHelpOptions = true,
    sortOptions = false,
    footer = {
        "",
        "Metrics Computed:",
        "  Accuracy     Overall correct predictions / total predictions",
        "  Precision    True positives / (true positives + false positives)",
        "  Recall       True positives / (true positives + false negatives)",
        "  F1 Score     Harmonic mean of precision and recall",
        "",
        "Examples:",
        "  @|bold face-recognition benchmark -d lfw/ -a eigenfaces --folds 5|@",
        "  @|bold face-recognition benchmark -d dataset/ --compare-all|@"
    }
)
public class BenchmarkCommand implements Callable<Integer> {

    @ParentCommand
    private FaceRecognitionCLI parent;

    @Parameters(index = "0", arity = "0..1",
            description = "Dataset directory (alternative to -d)")
    private File datasetParam;

    @Option(names = {"-d", "--dataset"}, paramLabel = "DIR",
            description = "Path to the benchmark dataset directory")
    private File datasetDir;

    @Option(names = {"-a", "--algorithm"}, paramLabel = "ALG",
            description = "Algorithm to benchmark: ${COMPLETION-CANDIDATES} (default: ${DEFAULT-VALUE})",
            defaultValue = "eigenfaces")
    private Algorithm algorithm;

    @Option(names = {"--compare-all"},
            description = "Compare all available algorithms")
    private boolean compareAll;

    @Option(names = {"-k", "--folds"}, paramLabel = "K",
            description = "Number of cross-validation folds (default: ${DEFAULT-VALUE})",
            defaultValue = "5")
    private int folds;

    @Option(names = {"--components"}, paramLabel = "N",
            description = "Number of components for PCA-based algorithms (default: ${DEFAULT-VALUE})",
            defaultValue = "10")
    private int components;

    @Option(names = {"-t", "--threshold"}, paramLabel = "VALUE",
            description = "Recognition threshold (default: ${DEFAULT-VALUE})",
            defaultValue = "0.6")
    private double threshold;

    @Option(names = {"-o", "--output"}, paramLabel = "FILE",
            description = "Output file for detailed results (JSON format)")
    private File outputFile;

    @Option(names = {"--min-samples"}, paramLabel = "N",
            description = "Minimum samples per identity (default: ${DEFAULT-VALUE})",
            defaultValue = "2")
    private int minSamples;

    @Option(names = {"--max-identities"}, paramLabel = "N",
            description = "Maximum identities to use (0 = all, default: ${DEFAULT-VALUE})",
            defaultValue = "0")
    private int maxIdentities;

    @Option(names = {"-v", "--verbose"},
            description = "Show detailed progress and per-fold results")
    private boolean verbose;

    @Option(names = {"--confusion-matrix"},
            description = "Generate confusion matrix")
    private boolean confusionMatrix;

    @Option(names = {"--extensions"}, paramLabel = "EXT",
            description = "File extensions to process (default: jpg,jpeg,png,bmp)",
            split = ",",
            defaultValue = "jpg,jpeg,png,bmp")
    private Set<String> extensions;

    @Option(names = {"--seed"}, paramLabel = "N",
            description = "Random seed for reproducibility (default: ${DEFAULT-VALUE})",
            defaultValue = "42")
    private long seed;

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
        if (folds < 2) {
            System.err.println("Error: Number of folds must be at least 2");
            return FaceRecognitionCLI.EXIT_INVALID_ARGS;
        }

        // Collect dataset
        if (!isQuiet()) {
            System.out.println("Loading dataset from: " + dataset);
        }

        Map<String, List<File>> identityFiles = collectDataset(dataset);
        identityFiles = filterByMinSamples(identityFiles);

        if (identityFiles.isEmpty()) {
            System.err.println("Error: No valid identities found (minimum " + minSamples + " samples required)");
            return FaceRecognitionCLI.EXIT_FILE_NOT_FOUND;
        }

        // Limit identities if requested
        if (maxIdentities > 0 && identityFiles.size() > maxIdentities) {
            identityFiles = new LinkedHashMap<>(identityFiles.entrySet().stream()
                .limit(maxIdentities)
                .collect(Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue)));
        }

        int totalFiles = identityFiles.values().stream().mapToInt(List::size).sum();
        int totalIdentities = identityFiles.size();

        if (!isQuiet()) {
            System.out.println("Dataset: " + totalFiles + " images, " + totalIdentities + " identities");
            System.out.println("Folds: " + folds);
            System.out.println("Threshold: " + threshold);
            System.out.println();
        }

        // Run benchmarks
        List<BenchmarkResult> results = new ArrayList<>();

        if (compareAll) {
            for (Algorithm alg : Algorithm.values()) {
                if (!isQuiet()) {
                    System.out.println("=" .repeat(60));
                    System.out.println("BENCHMARKING: " + alg.name().toUpperCase());
                    System.out.println("=".repeat(60));
                }
                BenchmarkResult result = runBenchmark(alg, identityFiles);
                results.add(result);
            }
        } else {
            BenchmarkResult result = runBenchmark(algorithm, identityFiles);
            results.add(result);
        }

        // Print summary
        printSummary(results);

        // Save detailed results if requested
        if (outputFile != null) {
            saveResults(results, totalIdentities, totalFiles);
        }

        return FaceRecognitionCLI.EXIT_SUCCESS;
    }

    private boolean isQuiet() {
        return parent != null && parent.isQuiet();
    }

    private Map<String, List<File>> collectDataset(File dataset) {
        Map<String, List<File>> result = new LinkedHashMap<>();

        File[] subdirs = dataset.listFiles(File::isDirectory);
        if (subdirs == null) return result;

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

    private Map<String, List<File>> filterByMinSamples(Map<String, List<File>> data) {
        return data.entrySet().stream()
            .filter(e -> e.getValue().size() >= minSamples)
            .collect(Collectors.toMap(
                Map.Entry::getKey,
                Map.Entry::getValue,
                (a, b) -> a,
                LinkedHashMap::new
            ));
    }

    private boolean isImageFile(File file) {
        String name = file.getName().toLowerCase();
        return extensions.stream().anyMatch(ext -> name.endsWith("." + ext.toLowerCase()));
    }

    private BenchmarkResult runBenchmark(Algorithm alg, Map<String, List<File>> identityFiles) {
        BenchmarkResult result = new BenchmarkResult();
        result.algorithm = alg.name();
        result.components = components;
        result.threshold = threshold;
        result.folds = folds;
        result.startTime = System.currentTimeMillis();
        result.foldResults = new ArrayList<>();

        // Create fold assignments
        List<FoldData> foldDataList = createFolds(identityFiles);

        // Run each fold
        for (int fold = 0; fold < folds; fold++) {
            if (!isQuiet()) {
                System.out.printf("  Fold %d/%d: ", fold + 1, folds);
            }

            FoldResult foldResult = runFold(alg, foldDataList, fold);
            result.foldResults.add(foldResult);

            if (!isQuiet()) {
                System.out.printf("Accuracy: %.2f%%, Precision: %.2f%%, Recall: %.2f%%%n",
                    foldResult.accuracy * 100, foldResult.precision * 100, foldResult.recall * 100);
            }

            if (verbose) {
                System.out.printf("    TP: %d, FP: %d, TN: %d, FN: %d%n",
                    foldResult.truePositives, foldResult.falsePositives,
                    foldResult.trueNegatives, foldResult.falseNegatives);
            }
        }

        result.endTime = System.currentTimeMillis();

        // Compute aggregate metrics
        computeAggregateMetrics(result);

        // Build confusion matrix if requested
        if (confusionMatrix) {
            result.confusionMatrix = buildConfusionMatrix(identityFiles.keySet(), result.foldResults);
        }

        return result;
    }

    private List<FoldData> createFolds(Map<String, List<File>> identityFiles) {
        List<FoldData> foldDataList = new ArrayList<>();
        Random random = new Random(seed);

        for (int i = 0; i < folds; i++) {
            foldDataList.add(new FoldData());
        }

        // Stratified split - distribute samples from each identity across folds
        for (Map.Entry<String, List<File>> entry : identityFiles.entrySet()) {
            String identity = entry.getKey();
            List<File> files = new ArrayList<>(entry.getValue());
            Collections.shuffle(files, random);

            for (int i = 0; i < files.size(); i++) {
                int foldIdx = i % folds;
                foldDataList.get(foldIdx).samples.add(new Sample(files.get(i), identity));
            }
        }

        return foldDataList;
    }

    private FoldResult runFold(Algorithm alg, List<FoldData> foldDataList, int testFoldIdx) {
        FoldResult result = new FoldResult();
        result.foldIndex = testFoldIdx;

        // Collect training and test data
        List<Sample> trainSamples = new ArrayList<>();
        List<Sample> testSamples = new ArrayList<>();

        for (int i = 0; i < folds; i++) {
            if (i == testFoldIdx) {
                testSamples.addAll(foldDataList.get(i).samples);
            } else {
                trainSamples.addAll(foldDataList.get(i).samples);
            }
        }

        // Load training images
        List<FaceImage> trainImages = new ArrayList<>();
        List<String> trainLabels = new ArrayList<>();

        for (Sample sample : trainSamples) {
            try {
                trainImages.add(FaceImage.fromFile(sample.file));
                trainLabels.add(sample.identity);
            } catch (Exception e) {
                // Skip invalid files
            }
        }

        if (trainImages.isEmpty()) {
            result.error = "No valid training images";
            return result;
        }

        // Create and train extractor
        FeatureExtractor extractor = createExtractor(alg);
        FaceClassifier classifier = new KNNClassifier();

        try {
            extractor.train(trainImages, trainLabels);
        } catch (Exception e) {
            result.error = "Training failed: " + e.getMessage();
            return result;
        }

        // Build service
        FaceRecognitionService service = FaceRecognitionService.builder()
            .extractor(extractor)
            .classifier(classifier)
            .build();

        // Enroll training samples
        for (int i = 0; i < trainImages.size(); i++) {
            service.enroll(trainImages.get(i), trainLabels.get(i));
        }
        service.train();

        // Test
        result.predictions = new ArrayList<>();
        int correct = 0;
        int total = 0;

        for (Sample sample : testSamples) {
            try {
                FaceImage testImage = FaceImage.fromFile(sample.file);
                RecognitionResult recognition = service.recognize(testImage);

                Prediction pred = new Prediction();
                pred.actual = sample.identity;
                pred.file = sample.file.getName();

                if (recognition.isRecognized()) {
                    pred.predicted = recognition.getBestMatch().get().getIdentity().getName();
                    pred.confidence = recognition.getBestMatch().get().getConfidence();

                    if (pred.predicted.equals(sample.identity)) {
                        correct++;
                        result.truePositives++;
                    } else {
                        result.falsePositives++;
                    }
                } else {
                    pred.predicted = null;
                    pred.confidence = 0;
                    result.falseNegatives++;
                }

                result.predictions.add(pred);
                total++;

            } catch (Exception e) {
                // Skip invalid test files
            }
        }

        // Compute metrics
        result.total = total;
        result.correct = correct;
        result.accuracy = total > 0 ? (double) correct / total : 0;

        // Precision = TP / (TP + FP)
        int tpfp = result.truePositives + result.falsePositives;
        result.precision = tpfp > 0 ? (double) result.truePositives / tpfp : 0;

        // Recall = TP / (TP + FN)
        int tpfn = result.truePositives + result.falseNegatives;
        result.recall = tpfn > 0 ? (double) result.truePositives / tpfn : 0;

        // F1 = 2 * (precision * recall) / (precision + recall)
        double prSum = result.precision + result.recall;
        result.f1Score = prSum > 0 ? 2 * result.precision * result.recall / prSum : 0;

        return result;
    }

    private FeatureExtractor createExtractor(Algorithm alg) {
        FeatureExtractor.ExtractorConfig config = new FeatureExtractor.ExtractorConfig()
            .setNumComponents(components)
            .setNormalize(true);

        switch (alg) {
            case lbph:
                return new LBPHExtractor();
            case fisherfaces:
                return new FisherfacesExtractor(config);
            case eigenfaces:
            default:
                return new EigenfacesExtractor(config);
        }
    }

    private void computeAggregateMetrics(BenchmarkResult result) {
        // Average across folds
        result.accuracy = result.foldResults.stream()
            .mapToDouble(f -> f.accuracy).average().orElse(0);
        result.precision = result.foldResults.stream()
            .mapToDouble(f -> f.precision).average().orElse(0);
        result.recall = result.foldResults.stream()
            .mapToDouble(f -> f.recall).average().orElse(0);
        result.f1Score = result.foldResults.stream()
            .mapToDouble(f -> f.f1Score).average().orElse(0);

        // Standard deviations
        result.accuracyStd = computeStdDev(result.foldResults.stream()
            .mapToDouble(f -> f.accuracy).toArray(), result.accuracy);
        result.precisionStd = computeStdDev(result.foldResults.stream()
            .mapToDouble(f -> f.precision).toArray(), result.precision);
        result.recallStd = computeStdDev(result.foldResults.stream()
            .mapToDouble(f -> f.recall).toArray(), result.recall);
        result.f1ScoreStd = computeStdDev(result.foldResults.stream()
            .mapToDouble(f -> f.f1Score).toArray(), result.f1Score);
    }

    private double computeStdDev(double[] values, double mean) {
        double sumSq = 0;
        for (double v : values) {
            sumSq += (v - mean) * (v - mean);
        }
        return Math.sqrt(sumSq / values.length);
    }

    private int[][] buildConfusionMatrix(Set<String> identities, List<FoldResult> foldResults) {
        List<String> labels = new ArrayList<>(identities);
        int n = labels.size();
        int[][] matrix = new int[n][n];

        Map<String, Integer> labelIndex = new HashMap<>();
        for (int i = 0; i < n; i++) {
            labelIndex.put(labels.get(i), i);
        }

        for (FoldResult fold : foldResults) {
            for (Prediction pred : fold.predictions) {
                if (pred.predicted != null && labelIndex.containsKey(pred.predicted)) {
                    int actualIdx = labelIndex.get(pred.actual);
                    int predIdx = labelIndex.get(pred.predicted);
                    matrix[actualIdx][predIdx]++;
                }
            }
        }

        return matrix;
    }

    private void printSummary(List<BenchmarkResult> results) {
        System.out.println("\n" + "=".repeat(70));
        System.out.println("BENCHMARK RESULTS");
        System.out.println("=".repeat(70));

        // Print header
        System.out.printf("%-15s %12s %12s %12s %12s%n",
            "Algorithm", "Accuracy", "Precision", "Recall", "F1 Score");
        System.out.println("-".repeat(70));

        // Print results
        for (BenchmarkResult result : results) {
            System.out.printf("%-15s %10.2f%% %10.2f%% %10.2f%% %10.2f%%%n",
                result.algorithm,
                result.accuracy * 100,
                result.precision * 100,
                result.recall * 100,
                result.f1Score * 100);

            if (verbose) {
                System.out.printf("               %11s %11s %11s %11s%n",
                    String.format("+/- %.2f%%", result.accuracyStd * 100),
                    String.format("+/- %.2f%%", result.precisionStd * 100),
                    String.format("+/- %.2f%%", result.recallStd * 100),
                    String.format("+/- %.2f%%", result.f1ScoreStd * 100));
            }
        }

        System.out.println("-".repeat(70));

        // Find best
        if (results.size() > 1) {
            BenchmarkResult best = results.stream()
                .max(Comparator.comparingDouble(r -> r.f1Score))
                .orElse(results.get(0));
            System.out.println("\nBest algorithm by F1 Score: " + best.algorithm);
        }

        // Timing
        for (BenchmarkResult result : results) {
            long time = result.endTime - result.startTime;
            System.out.printf("%s total time: %s%n", result.algorithm,
                FaceRecognitionCLI.formatDuration(time));
        }
    }

    private void saveResults(List<BenchmarkResult> results, int numIdentities, int numImages) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.registerModule(new JavaTimeModule());
        mapper.enable(SerializationFeature.INDENT_OUTPUT);

        Map<String, Object> output = new LinkedHashMap<>();
        output.put("timestamp", LocalDateTime.now().toString());
        output.put("numIdentities", numIdentities);
        output.put("numImages", numImages);
        output.put("folds", folds);
        output.put("threshold", threshold);
        output.put("seed", seed);

        List<Map<String, Object>> benchmarks = new ArrayList<>();
        for (BenchmarkResult result : results) {
            Map<String, Object> bench = new LinkedHashMap<>();
            bench.put("algorithm", result.algorithm);
            bench.put("components", result.components);
            bench.put("accuracy", result.accuracy);
            bench.put("accuracyStd", result.accuracyStd);
            bench.put("precision", result.precision);
            bench.put("precisionStd", result.precisionStd);
            bench.put("recall", result.recall);
            bench.put("recallStd", result.recallStd);
            bench.put("f1Score", result.f1Score);
            bench.put("f1ScoreStd", result.f1ScoreStd);
            bench.put("durationMs", result.endTime - result.startTime);

            if (verbose) {
                List<Map<String, Object>> foldDetails = new ArrayList<>();
                for (FoldResult fold : result.foldResults) {
                    Map<String, Object> f = new LinkedHashMap<>();
                    f.put("fold", fold.foldIndex);
                    f.put("accuracy", fold.accuracy);
                    f.put("precision", fold.precision);
                    f.put("recall", fold.recall);
                    f.put("f1Score", fold.f1Score);
                    f.put("truePositives", fold.truePositives);
                    f.put("falsePositives", fold.falsePositives);
                    f.put("falseNegatives", fold.falseNegatives);
                    foldDetails.add(f);
                }
                bench.put("foldDetails", foldDetails);
            }

            if (result.confusionMatrix != null) {
                bench.put("confusionMatrix", result.confusionMatrix);
            }

            benchmarks.add(bench);
        }

        output.put("benchmarks", benchmarks);

        mapper.writeValue(outputFile, output);
        System.out.println("\nDetailed results saved to: " + outputFile);
    }

    // Inner classes for data structures
    private static class Sample {
        File file;
        String identity;

        Sample(File file, String identity) {
            this.file = file;
            this.identity = identity;
        }
    }

    private static class FoldData {
        List<Sample> samples = new ArrayList<>();
    }

    private static class Prediction {
        String actual;
        String predicted;
        double confidence;
        String file;
    }

    private static class FoldResult {
        int foldIndex;
        int total;
        int correct;
        double accuracy;
        double precision;
        double recall;
        double f1Score;
        int truePositives;
        int falsePositives;
        int trueNegatives;
        int falseNegatives;
        List<Prediction> predictions;
        String error;
    }

    private static class BenchmarkResult {
        String algorithm;
        int components;
        double threshold;
        int folds;
        double accuracy;
        double accuracyStd;
        double precision;
        double precisionStd;
        double recall;
        double recallStd;
        double f1Score;
        double f1ScoreStd;
        long startTime;
        long endTime;
        List<FoldResult> foldResults;
        int[][] confusionMatrix;
    }
}
