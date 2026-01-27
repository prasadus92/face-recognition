package com.facerecognition.benchmark;

import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.domain.service.FaceClassifier;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.infrastructure.extraction.FisherfacesExtractor;
import com.facerecognition.infrastructure.extraction.LBPHExtractor;
import com.facerecognition.infrastructure.classification.KNNClassifier;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;
import java.util.function.Supplier;

/**
 * Main benchmark runner for the face recognition system.
 *
 * <p>This class provides a command-line interface and programmatic API for running
 * comprehensive benchmarks on face recognition algorithms. It supports:</p>
 * <ul>
 *   <li>Running accuracy benchmarks with various metrics</li>
 *   <li>Running performance/timing benchmarks</li>
 *   <li>Cross-validation with multiple strategies</li>
 *   <li>Comparing multiple algorithms</li>
 *   <li>Generating reports in multiple formats (HTML, JSON, Markdown, CSV)</li>
 * </ul>
 *
 * <h3>Command Line Usage:</h3>
 * <pre>
 * java -cp ... com.facerecognition.benchmark.BenchmarkRunner [options]
 *
 * Options:
 *   --dataset PATH       Path to dataset directory
 *   --format FORMAT      Dataset format: orl, yale, lfw, custom (default: custom)
 *   --output DIR         Output directory for reports (default: ./benchmark_results)
 *   --algorithms NAMES   Comma-separated list of algorithms: eigenfaces,fisherfaces,lbph (default: all)
 *   --type TYPE          Benchmark type: accuracy, performance, crossval, all (default: all)
 *   --folds N            Number of folds for cross-validation (default: 5)
 *   --train-ratio R      Training ratio for train/test split (default: 0.7)
 *   --threshold T        Recognition threshold (default: 0.6)
 *   --warmup N           Warmup iterations for performance benchmark (default: 10)
 *   --runs N             Measurement runs for performance benchmark (default: 100)
 *   --components N       Number of components for PCA/LDA (default: 10)
 *   --verbose            Enable verbose output
 *   --help               Show this help message
 * </pre>
 *
 * <h3>Programmatic Usage:</h3>
 * <pre>{@code
 * BenchmarkRunner runner = BenchmarkRunner.builder()
 *     .datasetPath("/path/to/dataset")
 *     .datasetFormat("orl")
 *     .outputDirectory(Paths.get("./results"))
 *     .addAlgorithm("Eigenfaces", () -> new EigenfacesExtractor(10))
 *     .addAlgorithm("LBPH", () -> new LBPHExtractor())
 *     .benchmarkType(BenchmarkRunner.BenchmarkType.ALL)
 *     .build();
 *
 * BenchmarkSuite results = runner.run();
 * System.out.println(results.getSummary());
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see AccuracyBenchmark
 * @see PerformanceBenchmark
 * @see CrossValidator
 */
public class BenchmarkRunner {

    private static final DateTimeFormatter TIMESTAMP_FORMAT =
        DateTimeFormatter.ofPattern("yyyyMMdd_HHmmss");

    /**
     * Benchmark types that can be run.
     */
    public enum BenchmarkType {
        /** Accuracy benchmarks only. */
        ACCURACY,
        /** Performance benchmarks only. */
        PERFORMANCE,
        /** Cross-validation benchmarks only. */
        CROSS_VALIDATION,
        /** All benchmark types. */
        ALL
    }

    private final Path datasetPath;
    private final String datasetFormat;
    private final Path outputDirectory;
    private final Map<String, Supplier<FeatureExtractor>> algorithms;
    private final Supplier<FaceClassifier> classifierSupplier;
    private final BenchmarkType benchmarkType;
    private final RunnerConfig config;

    /**
     * Runner configuration options.
     */
    public static class RunnerConfig {
        private int crossValFolds = 5;
        private double trainRatio = 0.7;
        private double recognitionThreshold = 0.6;
        private int warmupRuns = 10;
        private int measurementRuns = 100;
        private int numComponents = 10;
        private long randomSeed = 42;
        private boolean verbose = true;
        private boolean generateHtml = true;
        private boolean generateJson = true;
        private boolean generateMarkdown = true;
        private boolean generateCsv = true;
        private int targetWidth = 48;
        private int targetHeight = 64;

        public int getCrossValFolds() { return crossValFolds; }
        public RunnerConfig setCrossValFolds(int f) { this.crossValFolds = f; return this; }

        public double getTrainRatio() { return trainRatio; }
        public RunnerConfig setTrainRatio(double r) { this.trainRatio = r; return this; }

        public double getRecognitionThreshold() { return recognitionThreshold; }
        public RunnerConfig setRecognitionThreshold(double t) { this.recognitionThreshold = t; return this; }

        public int getWarmupRuns() { return warmupRuns; }
        public RunnerConfig setWarmupRuns(int w) { this.warmupRuns = w; return this; }

        public int getMeasurementRuns() { return measurementRuns; }
        public RunnerConfig setMeasurementRuns(int m) { this.measurementRuns = m; return this; }

        public int getNumComponents() { return numComponents; }
        public RunnerConfig setNumComponents(int n) { this.numComponents = n; return this; }

        public long getRandomSeed() { return randomSeed; }
        public RunnerConfig setRandomSeed(long s) { this.randomSeed = s; return this; }

        public boolean isVerbose() { return verbose; }
        public RunnerConfig setVerbose(boolean v) { this.verbose = v; return this; }

        public boolean isGenerateHtml() { return generateHtml; }
        public RunnerConfig setGenerateHtml(boolean g) { this.generateHtml = g; return this; }

        public boolean isGenerateJson() { return generateJson; }
        public RunnerConfig setGenerateJson(boolean g) { this.generateJson = g; return this; }

        public boolean isGenerateMarkdown() { return generateMarkdown; }
        public RunnerConfig setGenerateMarkdown(boolean g) { this.generateMarkdown = g; return this; }

        public boolean isGenerateCsv() { return generateCsv; }
        public RunnerConfig setGenerateCsv(boolean g) { this.generateCsv = g; return this; }

        public int getTargetWidth() { return targetWidth; }
        public RunnerConfig setTargetWidth(int w) { this.targetWidth = w; return this; }

        public int getTargetHeight() { return targetHeight; }
        public RunnerConfig setTargetHeight(int h) { this.targetHeight = h; return this; }
    }

    /**
     * Results of a complete benchmark suite run.
     */
    public static class BenchmarkSuite {
        private final LocalDateTime runTime;
        private final DatasetLoader.LoadedDataset dataset;
        private final Map<String, BenchmarkResult> accuracyResults;
        private final Map<String, BenchmarkResult> performanceResults;
        private final Map<String, CrossValidator.CrossValidationResult> crossValResults;
        private final Path outputDirectory;
        private final long totalDurationMs;

        public BenchmarkSuite(LocalDateTime runTime, DatasetLoader.LoadedDataset dataset,
                             Map<String, BenchmarkResult> accuracyResults,
                             Map<String, BenchmarkResult> performanceResults,
                             Map<String, CrossValidator.CrossValidationResult> crossValResults,
                             Path outputDirectory, long totalDurationMs) {
            this.runTime = runTime;
            this.dataset = dataset;
            this.accuracyResults = new LinkedHashMap<>(accuracyResults);
            this.performanceResults = new LinkedHashMap<>(performanceResults);
            this.crossValResults = new LinkedHashMap<>(crossValResults);
            this.outputDirectory = outputDirectory;
            this.totalDurationMs = totalDurationMs;
        }

        public LocalDateTime getRunTime() { return runTime; }
        public DatasetLoader.LoadedDataset getDataset() { return dataset; }
        public Map<String, BenchmarkResult> getAccuracyResults() { return Collections.unmodifiableMap(accuracyResults); }
        public Map<String, BenchmarkResult> getPerformanceResults() { return Collections.unmodifiableMap(performanceResults); }
        public Map<String, CrossValidator.CrossValidationResult> getCrossValResults() { return Collections.unmodifiableMap(crossValResults); }
        public Path getOutputDirectory() { return outputDirectory; }
        public long getTotalDurationMs() { return totalDurationMs; }

        /**
         * Gets all benchmark results combined.
         *
         * @return combined map of all results
         */
        public Map<String, BenchmarkResult> getAllResults() {
            Map<String, BenchmarkResult> all = new LinkedHashMap<>();
            all.putAll(accuracyResults);
            all.putAll(performanceResults);
            for (Map.Entry<String, CrossValidator.CrossValidationResult> entry : crossValResults.entrySet()) {
                all.put(entry.getKey() + " (CV)",
                    entry.getValue().toBenchmarkResult(entry.getKey(), "CrossVal"));
            }
            return all;
        }

        /**
         * Generates a summary of the benchmark suite.
         *
         * @return formatted summary string
         */
        public String getSummary() {
            StringBuilder sb = new StringBuilder();

            sb.append("=".repeat(60)).append("\n");
            sb.append("           BENCHMARK SUITE SUMMARY\n");
            sb.append("=".repeat(60)).append("\n\n");

            sb.append(String.format("Run Time: %s%n", runTime.format(DateTimeFormatter.ISO_LOCAL_DATE_TIME)));
            sb.append(String.format("Total Duration: %.2f seconds%n%n", totalDurationMs / 1000.0));

            sb.append("Dataset: ").append(dataset.getName()).append("\n");
            sb.append(String.format("  Classes: %d, Images: %d%n%n",
                dataset.getNumClasses(), dataset.getFaces().size()));

            if (!accuracyResults.isEmpty()) {
                sb.append("--- ACCURACY RESULTS ---\n");
                sb.append(String.format("%-20s %10s %10s %10s%n", "Algorithm", "Accuracy", "F1-Score", "EER"));
                sb.append("-".repeat(52)).append("\n");
                for (Map.Entry<String, BenchmarkResult> entry : accuracyResults.entrySet()) {
                    BenchmarkResult r = entry.getValue();
                    sb.append(String.format("%-20s %10.4f %10.4f %10.4f%n",
                        truncate(entry.getKey(), 20), r.getAccuracy(), r.getF1Score(), r.getEqualErrorRate()));
                }
                sb.append("\n");
            }

            if (!crossValResults.isEmpty()) {
                sb.append("--- CROSS-VALIDATION RESULTS ---\n");
                sb.append(String.format("%-20s %10s %10s %10s%n", "Algorithm", "Accuracy", "Std Dev", "95% CI"));
                sb.append("-".repeat(52)).append("\n");
                for (Map.Entry<String, CrossValidator.CrossValidationResult> entry : crossValResults.entrySet()) {
                    CrossValidator.CrossValidationResult r = entry.getValue();
                    double[] ci = r.getAccuracyConfidenceInterval();
                    sb.append(String.format("%-20s %10.4f %10.4f [%.3f-%.3f]%n",
                        truncate(entry.getKey(), 20), r.getMeanAccuracy(), r.getStdAccuracy(), ci[0], ci[1]));
                }
                sb.append("\n");
            }

            if (!performanceResults.isEmpty()) {
                sb.append("--- PERFORMANCE RESULTS ---\n");
                sb.append(String.format("%-20s %12s %12s %12s%n", "Algorithm", "Throughput", "Avg Time", "Peak Mem"));
                sb.append("-".repeat(58)).append("\n");
                for (Map.Entry<String, BenchmarkResult> entry : performanceResults.entrySet()) {
                    BenchmarkResult r = entry.getValue();
                    r.getPerformanceMetrics().ifPresent(m -> {
                        double avgTime = m.getTotalTime() != null ? m.getTotalTime().getMean() : 0;
                        sb.append(String.format("%-20s %10.2f/s %10.2fms %10.2fMB%n",
                            truncate(entry.getKey(), 20), m.getThroughput(), avgTime, m.getPeakMemoryMB()));
                    });
                }
                sb.append("\n");
            }

            sb.append("Output Directory: ").append(outputDirectory).append("\n");
            sb.append("=".repeat(60)).append("\n");

            return sb.toString();
        }

        private static String truncate(String s, int maxLen) {
            return s.length() <= maxLen ? s : s.substring(0, maxLen - 2) + "..";
        }
    }

    /**
     * Builder for BenchmarkRunner.
     */
    public static class Builder {
        private Path datasetPath;
        private String datasetFormat = "custom";
        private Path outputDirectory = Paths.get("./benchmark_results");
        private Map<String, Supplier<FeatureExtractor>> algorithms = new LinkedHashMap<>();
        private Supplier<FaceClassifier> classifierSupplier = KNNClassifier::new;
        private BenchmarkType benchmarkType = BenchmarkType.ALL;
        private RunnerConfig config = new RunnerConfig();

        public Builder datasetPath(String path) {
            this.datasetPath = Paths.get(path);
            return this;
        }

        public Builder datasetPath(Path path) {
            this.datasetPath = path;
            return this;
        }

        public Builder datasetFormat(String format) {
            this.datasetFormat = format.toLowerCase();
            return this;
        }

        public Builder outputDirectory(Path path) {
            this.outputDirectory = path;
            return this;
        }

        public Builder outputDirectory(String path) {
            this.outputDirectory = Paths.get(path);
            return this;
        }

        public Builder addAlgorithm(String name, Supplier<FeatureExtractor> supplier) {
            this.algorithms.put(name, supplier);
            return this;
        }

        public Builder classifierSupplier(Supplier<FaceClassifier> supplier) {
            this.classifierSupplier = supplier;
            return this;
        }

        public Builder benchmarkType(BenchmarkType type) {
            this.benchmarkType = type;
            return this;
        }

        public Builder config(RunnerConfig config) {
            this.config = config;
            return this;
        }

        /**
         * Adds default algorithms for benchmarking.
         *
         * @return this builder
         */
        public Builder addDefaultAlgorithms() {
            int components = config.getNumComponents();
            algorithms.put("Eigenfaces", () -> new EigenfacesExtractor(components));
            algorithms.put("Fisherfaces", () -> new FisherfacesExtractor(components));
            algorithms.put("LBPH", LBPHExtractor::new);
            return this;
        }

        public BenchmarkRunner build() {
            Objects.requireNonNull(datasetPath, "Dataset path is required");
            if (algorithms.isEmpty()) {
                addDefaultAlgorithms();
            }
            return new BenchmarkRunner(datasetPath, datasetFormat, outputDirectory,
                algorithms, classifierSupplier, benchmarkType, config);
        }
    }

    private BenchmarkRunner(Path datasetPath, String datasetFormat, Path outputDirectory,
                            Map<String, Supplier<FeatureExtractor>> algorithms,
                            Supplier<FaceClassifier> classifierSupplier,
                            BenchmarkType benchmarkType, RunnerConfig config) {
        this.datasetPath = datasetPath;
        this.datasetFormat = datasetFormat;
        this.outputDirectory = outputDirectory;
        this.algorithms = new LinkedHashMap<>(algorithms);
        this.classifierSupplier = classifierSupplier;
        this.benchmarkType = benchmarkType;
        this.config = config;
    }

    /**
     * Creates a new builder.
     *
     * @return a new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    /**
     * Runs the benchmark suite.
     *
     * @return benchmark suite results
     * @throws IOException if loading dataset or saving reports fails
     */
    public BenchmarkSuite run() throws IOException {
        LocalDateTime startTime = LocalDateTime.now();
        long startMs = System.currentTimeMillis();

        // Print header
        if (config.isVerbose()) {
            printHeader();
        }

        // Load dataset
        DatasetLoader.LoadedDataset dataset = loadDataset();

        // Create output directory
        String timestamp = startTime.format(TIMESTAMP_FORMAT);
        Path runOutputDir = outputDirectory.resolve(timestamp);
        Files.createDirectories(runOutputDir);

        // Initialize result containers
        Map<String, BenchmarkResult> accuracyResults = new LinkedHashMap<>();
        Map<String, BenchmarkResult> performanceResults = new LinkedHashMap<>();
        Map<String, CrossValidator.CrossValidationResult> crossValResults = new LinkedHashMap<>();

        // Run benchmarks
        if (benchmarkType == BenchmarkType.ACCURACY || benchmarkType == BenchmarkType.ALL) {
            if (config.isVerbose()) {
                System.out.println("\n=== RUNNING ACCURACY BENCHMARKS ===\n");
            }
            accuracyResults = runAccuracyBenchmarks(dataset);
        }

        if (benchmarkType == BenchmarkType.CROSS_VALIDATION || benchmarkType == BenchmarkType.ALL) {
            if (config.isVerbose()) {
                System.out.println("\n=== RUNNING CROSS-VALIDATION ===\n");
            }
            crossValResults = runCrossValidation(dataset);
        }

        if (benchmarkType == BenchmarkType.PERFORMANCE || benchmarkType == BenchmarkType.ALL) {
            if (config.isVerbose()) {
                System.out.println("\n=== RUNNING PERFORMANCE BENCHMARKS ===\n");
            }
            performanceResults = runPerformanceBenchmarks(dataset);
        }

        // Generate reports
        if (config.isVerbose()) {
            System.out.println("\n=== GENERATING REPORTS ===\n");
        }
        generateReports(runOutputDir, accuracyResults, performanceResults, crossValResults, dataset);

        long endMs = System.currentTimeMillis();

        BenchmarkSuite suite = new BenchmarkSuite(startTime, dataset,
            accuracyResults, performanceResults, crossValResults,
            runOutputDir, endMs - startMs);

        if (config.isVerbose()) {
            System.out.println(suite.getSummary());
        }

        return suite;
    }

    /**
     * Loads the dataset based on configuration.
     */
    private DatasetLoader.LoadedDataset loadDataset() throws IOException {
        if (config.isVerbose()) {
            System.out.printf("Loading dataset from: %s (format: %s)%n", datasetPath, datasetFormat);
        }

        DatasetLoader loader = new DatasetLoader()
            .setTargetSize(config.getTargetWidth(), config.getTargetHeight())
            .setNormalizeSize(true);

        DatasetLoader.LoadedDataset dataset;

        switch (datasetFormat.toLowerCase()) {
            case "orl":
            case "att":
                dataset = loader.loadOrl(datasetPath);
                break;
            case "yale":
                dataset = loader.loadYale(datasetPath);
                break;
            case "lfw":
                dataset = loader.loadLfw(datasetPath);
                break;
            case "flat":
                dataset = loader.loadFlatDirectory(datasetPath, "_");
                break;
            case "custom":
            default:
                dataset = loader.loadCustomDirectory(datasetPath.toString());
                break;
        }

        if (config.isVerbose()) {
            System.out.printf("Loaded %d images across %d classes%n",
                dataset.getFaces().size(), dataset.getNumClasses());
        }

        return dataset;
    }

    /**
     * Runs accuracy benchmarks for all algorithms.
     */
    private Map<String, BenchmarkResult> runAccuracyBenchmarks(DatasetLoader.LoadedDataset dataset) {
        Map<String, BenchmarkResult> results = new LinkedHashMap<>();

        AccuracyBenchmark.BenchmarkConfig accConfig = new AccuracyBenchmark.BenchmarkConfig()
            .setTrainRatio(config.getTrainRatio())
            .setRecognitionThreshold(config.getRecognitionThreshold())
            .setRandomSeed(config.getRandomSeed())
            .setVerbose(config.isVerbose());

        for (Map.Entry<String, Supplier<FeatureExtractor>> entry : algorithms.entrySet()) {
            String name = entry.getKey();
            if (config.isVerbose()) {
                System.out.printf("Running accuracy benchmark: %s%n", name);
            }

            AccuracyBenchmark benchmark = AccuracyBenchmark.builder()
                .extractor(entry.getValue().get())
                .classifier(classifierSupplier.get())
                .dataset(dataset)
                .config(accConfig)
                .name(name + " Accuracy")
                .build();

            BenchmarkResult result = benchmark.run();
            results.put(name, result);

            if (config.isVerbose()) {
                System.out.printf("  -> Accuracy: %.4f, F1: %.4f%n%n",
                    result.getAccuracy(), result.getF1Score());
            }
        }

        return results;
    }

    /**
     * Runs cross-validation for all algorithms.
     */
    private Map<String, CrossValidator.CrossValidationResult> runCrossValidation(
            DatasetLoader.LoadedDataset dataset) {

        Map<String, CrossValidator.CrossValidationResult> results = new LinkedHashMap<>();

        CrossValidator.CrossValidationConfig cvConfig = new CrossValidator.CrossValidationConfig()
            .setVerbose(config.isVerbose())
            .setRecognitionThreshold(config.getRecognitionThreshold());

        for (Map.Entry<String, Supplier<FeatureExtractor>> entry : algorithms.entrySet()) {
            String name = entry.getKey();
            if (config.isVerbose()) {
                System.out.printf("Running %d-fold cross-validation: %s%n", config.getCrossValFolds(), name);
            }

            CrossValidator validator = new CrossValidator(
                entry.getValue(), classifierSupplier, cvConfig);

            CrossValidator.CrossValidationResult result =
                validator.stratifiedKFold(dataset, config.getCrossValFolds(), config.getRandomSeed());
            results.put(name, result);

            if (config.isVerbose()) {
                System.out.printf("  -> Accuracy: %.4f +/- %.4f%n%n",
                    result.getMeanAccuracy(), result.getStdAccuracy());
            }
        }

        return results;
    }

    /**
     * Runs performance benchmarks for all algorithms.
     */
    private Map<String, BenchmarkResult> runPerformanceBenchmarks(DatasetLoader.LoadedDataset dataset) {
        Map<String, BenchmarkResult> results = new LinkedHashMap<>();

        PerformanceBenchmark.PerformanceConfig perfConfig = new PerformanceBenchmark.PerformanceConfig()
            .setWarmupRuns(config.getWarmupRuns())
            .setMeasurementRuns(config.getMeasurementRuns())
            .setTrainRatio(config.getTrainRatio())
            .setRandomSeed(config.getRandomSeed())
            .setVerbose(config.isVerbose());

        for (Map.Entry<String, Supplier<FeatureExtractor>> entry : algorithms.entrySet()) {
            String name = entry.getKey();
            if (config.isVerbose()) {
                System.out.printf("Running performance benchmark: %s%n", name);
            }

            PerformanceBenchmark benchmark = PerformanceBenchmark.builder()
                .extractor(entry.getValue().get())
                .classifier(classifierSupplier.get())
                .dataset(dataset)
                .config(perfConfig)
                .name(name + " Performance")
                .build();

            BenchmarkResult result = benchmark.run();
            results.put(name, result);

            result.getPerformanceMetrics().ifPresent(m -> {
                if (config.isVerbose()) {
                    System.out.printf("  -> Throughput: %.2f/s, Avg time: %.2fms%n%n",
                        m.getThroughput(), m.getTotalTime() != null ? m.getTotalTime().getMean() : 0);
                }
            });
        }

        return results;
    }

    /**
     * Generates all report files.
     */
    private void generateReports(Path outputDir,
                                 Map<String, BenchmarkResult> accuracyResults,
                                 Map<String, BenchmarkResult> performanceResults,
                                 Map<String, CrossValidator.CrossValidationResult> crossValResults,
                                 DatasetLoader.LoadedDataset dataset) throws IOException {

        ReportGenerator generator = new ReportGenerator();

        // Combine all results for main reports
        Map<String, BenchmarkResult> allResults = new LinkedHashMap<>();
        allResults.putAll(accuracyResults);
        allResults.putAll(performanceResults);
        for (Map.Entry<String, CrossValidator.CrossValidationResult> entry : crossValResults.entrySet()) {
            allResults.put(entry.getKey() + " (CV)",
                entry.getValue().toBenchmarkResult(entry.getKey(), entry.getValue().getStrategyName()));
        }

        // Generate reports
        if (config.isGenerateHtml()) {
            generator.saveHtml(allResults, outputDir.resolve("benchmark_report.html"));
            if (config.isVerbose()) {
                System.out.println("  Generated: benchmark_report.html");
            }
        }

        if (config.isGenerateJson()) {
            generator.saveJson(allResults, outputDir.resolve("benchmark_results.json"));
            if (config.isVerbose()) {
                System.out.println("  Generated: benchmark_results.json");
            }
        }

        if (config.isGenerateMarkdown()) {
            generator.saveMarkdown(allResults, outputDir.resolve("benchmark_report.md"));
            if (config.isVerbose()) {
                System.out.println("  Generated: benchmark_report.md");
            }
        }

        if (config.isGenerateCsv()) {
            generator.saveCsv(allResults, outputDir.resolve("benchmark_results.csv"));
            if (config.isVerbose()) {
                System.out.println("  Generated: benchmark_results.csv");
            }
        }

        // Generate separate detailed reports for each category
        if (!accuracyResults.isEmpty()) {
            generator.saveJson(accuracyResults, outputDir.resolve("accuracy_results.json"));
        }
        if (!performanceResults.isEmpty()) {
            generator.saveJson(performanceResults, outputDir.resolve("performance_results.json"));
        }
    }

    /**
     * Prints the benchmark header.
     */
    private void printHeader() {
        System.out.println();
        System.out.println("=".repeat(60));
        System.out.println("     Face Recognition Benchmark Suite v2.0");
        System.out.println("=".repeat(60));
        System.out.println();
        System.out.printf("Dataset:     %s%n", datasetPath);
        System.out.printf("Format:      %s%n", datasetFormat);
        System.out.printf("Output:      %s%n", outputDirectory);
        System.out.printf("Algorithms:  %s%n", String.join(", ", algorithms.keySet()));
        System.out.printf("Benchmark:   %s%n", benchmarkType);
        System.out.println();
    }

    // ==================== Command Line Interface ====================

    /**
     * Main entry point for command-line execution.
     *
     * @param args command-line arguments
     */
    public static void main(String[] args) {
        try {
            // Parse arguments
            CommandLineArgs cliArgs = parseArgs(args);

            if (cliArgs.showHelp) {
                printHelp();
                return;
            }

            if (cliArgs.datasetPath == null) {
                System.err.println("Error: --dataset path is required");
                printHelp();
                System.exit(1);
            }

            // Build runner
            Builder builder = builder()
                .datasetPath(cliArgs.datasetPath)
                .datasetFormat(cliArgs.datasetFormat)
                .outputDirectory(cliArgs.outputDirectory)
                .benchmarkType(cliArgs.benchmarkType);

            // Configure
            RunnerConfig config = new RunnerConfig()
                .setCrossValFolds(cliArgs.folds)
                .setTrainRatio(cliArgs.trainRatio)
                .setRecognitionThreshold(cliArgs.threshold)
                .setWarmupRuns(cliArgs.warmupRuns)
                .setMeasurementRuns(cliArgs.measurementRuns)
                .setNumComponents(cliArgs.components)
                .setVerbose(cliArgs.verbose);
            builder.config(config);

            // Add algorithms
            if (cliArgs.algorithms.isEmpty()) {
                builder.addDefaultAlgorithms();
            } else {
                for (String alg : cliArgs.algorithms) {
                    switch (alg.toLowerCase()) {
                        case "eigenfaces":
                            builder.addAlgorithm("Eigenfaces",
                                () -> new EigenfacesExtractor(cliArgs.components));
                            break;
                        case "fisherfaces":
                            builder.addAlgorithm("Fisherfaces",
                                () -> new FisherfacesExtractor(cliArgs.components));
                            break;
                        case "lbph":
                            builder.addAlgorithm("LBPH", LBPHExtractor::new);
                            break;
                        default:
                            System.err.println("Unknown algorithm: " + alg);
                    }
                }
            }

            // Run benchmarks
            BenchmarkRunner runner = builder.build();
            BenchmarkSuite results = runner.run();

            System.out.println("\nBenchmark suite completed successfully.");
            System.out.println("Reports saved to: " + results.getOutputDirectory());

        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }

    /**
     * Command-line arguments container.
     */
    private static class CommandLineArgs {
        String datasetPath;
        String datasetFormat = "custom";
        String outputDirectory = "./benchmark_results";
        List<String> algorithms = new ArrayList<>();
        BenchmarkType benchmarkType = BenchmarkType.ALL;
        int folds = 5;
        double trainRatio = 0.7;
        double threshold = 0.6;
        int warmupRuns = 10;
        int measurementRuns = 100;
        int components = 10;
        boolean verbose = true;
        boolean showHelp = false;
    }

    /**
     * Parses command-line arguments.
     */
    private static CommandLineArgs parseArgs(String[] args) {
        CommandLineArgs result = new CommandLineArgs();

        for (int i = 0; i < args.length; i++) {
            String arg = args[i];

            switch (arg) {
                case "--dataset":
                    result.datasetPath = args[++i];
                    break;
                case "--format":
                    result.datasetFormat = args[++i];
                    break;
                case "--output":
                    result.outputDirectory = args[++i];
                    break;
                case "--algorithms":
                    result.algorithms = Arrays.asList(args[++i].split(","));
                    break;
                case "--type":
                    result.benchmarkType = BenchmarkType.valueOf(args[++i].toUpperCase());
                    break;
                case "--folds":
                    result.folds = Integer.parseInt(args[++i]);
                    break;
                case "--train-ratio":
                    result.trainRatio = Double.parseDouble(args[++i]);
                    break;
                case "--threshold":
                    result.threshold = Double.parseDouble(args[++i]);
                    break;
                case "--warmup":
                    result.warmupRuns = Integer.parseInt(args[++i]);
                    break;
                case "--runs":
                    result.measurementRuns = Integer.parseInt(args[++i]);
                    break;
                case "--components":
                    result.components = Integer.parseInt(args[++i]);
                    break;
                case "--verbose":
                    result.verbose = true;
                    break;
                case "--quiet":
                    result.verbose = false;
                    break;
                case "--help":
                case "-h":
                    result.showHelp = true;
                    break;
                default:
                    if (arg.startsWith("-")) {
                        System.err.println("Unknown option: " + arg);
                    }
            }
        }

        return result;
    }

    /**
     * Prints help message.
     */
    private static void printHelp() {
        System.out.println(
            "Face Recognition Benchmark Runner\n" +
            "==================================\n\n" +
            "Usage: java -cp ... com.facerecognition.benchmark.BenchmarkRunner [options]\n\n" +
            "Required:\n" +
            "  --dataset PATH       Path to the dataset directory\n\n" +
            "Options:\n" +
            "  --format FORMAT      Dataset format: orl, yale, lfw, custom (default: custom)\n" +
            "  --output DIR         Output directory for reports (default: ./benchmark_results)\n" +
            "  --algorithms NAMES   Comma-separated list: eigenfaces,fisherfaces,lbph (default: all)\n" +
            "  --type TYPE          Benchmark type: accuracy, performance, cross_validation, all (default: all)\n" +
            "  --folds N            Number of folds for cross-validation (default: 5)\n" +
            "  --train-ratio R      Training ratio for train/test split (default: 0.7)\n" +
            "  --threshold T        Recognition threshold (default: 0.6)\n" +
            "  --warmup N           Warmup iterations for performance benchmark (default: 10)\n" +
            "  --runs N             Measurement runs for performance benchmark (default: 100)\n" +
            "  --components N       Number of components for PCA/LDA (default: 10)\n" +
            "  --verbose            Enable verbose output (default)\n" +
            "  --quiet              Disable verbose output\n" +
            "  --help, -h           Show this help message\n\n" +
            "Examples:\n" +
            "  # Run all benchmarks on ORL dataset\n" +
            "  java -cp ... BenchmarkRunner --dataset /path/to/orl_faces --format orl\n\n" +
            "  # Run only accuracy benchmark with specific algorithms\n" +
            "  java -cp ... BenchmarkRunner --dataset /data/faces --type accuracy --algorithms eigenfaces,lbph\n\n" +
            "  # Run cross-validation with 10 folds\n" +
            "  java -cp ... BenchmarkRunner --dataset /data/faces --type cross_validation --folds 10");
    }
}
