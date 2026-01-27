package com.facerecognition.benchmark;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.domain.service.FaceDetector;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.domain.service.FaceClassifier;

import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.*;
import java.util.function.Supplier;

/**
 * Performance benchmarking for face recognition algorithms.
 *
 * <p>This class measures various performance metrics:</p>
 * <ul>
 *   <li><b>Detection Time</b>: Time to detect faces in images</li>
 *   <li><b>Extraction Time</b>: Time to extract feature vectors</li>
 *   <li><b>Matching Time</b>: Time to classify/match features</li>
 *   <li><b>Memory Usage</b>: Peak and average memory consumption</li>
 *   <li><b>Throughput</b>: Samples processed per second</li>
 *   <li><b>Scalability</b>: Performance with increasing gallery size</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * PerformanceBenchmark benchmark = new PerformanceBenchmark.Builder()
 *     .extractor(new EigenfacesExtractor(10))
 *     .classifier(new KNNClassifier())
 *     .dataset(dataset)
 *     .warmupRuns(5)
 *     .measurementRuns(100)
 *     .build();
 *
 * BenchmarkResult result = benchmark.run();
 * System.out.println("Throughput: " + result.getPerformanceMetrics().get().getThroughput() + " fps");
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see BenchmarkResult
 */
public class PerformanceBenchmark {

    private final FaceDetector detector;
    private final FeatureExtractor extractor;
    private final FaceClassifier classifier;
    private final DatasetLoader.LoadedDataset dataset;
    private final PerformanceConfig config;

    private final MemoryMXBean memoryBean;

    /**
     * Configuration for performance benchmarks.
     */
    public static class PerformanceConfig {
        private int warmupRuns = 10;
        private int measurementRuns = 100;
        private boolean measureDetection = true;
        private boolean measureExtraction = true;
        private boolean measureMatching = true;
        private boolean measureMemory = true;
        private boolean measureThroughput = true;
        private boolean forceGc = true;
        private double trainRatio = 0.8;
        private long randomSeed = 42;
        private boolean verbose = true;
        private String benchmarkName = "Performance Benchmark";

        public int getWarmupRuns() { return warmupRuns; }
        public PerformanceConfig setWarmupRuns(int w) { this.warmupRuns = w; return this; }

        public int getMeasurementRuns() { return measurementRuns; }
        public PerformanceConfig setMeasurementRuns(int m) { this.measurementRuns = m; return this; }

        public boolean isMeasureDetection() { return measureDetection; }
        public PerformanceConfig setMeasureDetection(boolean m) { this.measureDetection = m; return this; }

        public boolean isMeasureExtraction() { return measureExtraction; }
        public PerformanceConfig setMeasureExtraction(boolean m) { this.measureExtraction = m; return this; }

        public boolean isMeasureMatching() { return measureMatching; }
        public PerformanceConfig setMeasureMatching(boolean m) { this.measureMatching = m; return this; }

        public boolean isMeasureMemory() { return measureMemory; }
        public PerformanceConfig setMeasureMemory(boolean m) { this.measureMemory = m; return this; }

        public boolean isMeasureThroughput() { return measureThroughput; }
        public PerformanceConfig setMeasureThroughput(boolean m) { this.measureThroughput = m; return this; }

        public boolean isForceGc() { return forceGc; }
        public PerformanceConfig setForceGc(boolean f) { this.forceGc = f; return this; }

        public double getTrainRatio() { return trainRatio; }
        public PerformanceConfig setTrainRatio(double r) { this.trainRatio = r; return this; }

        public long getRandomSeed() { return randomSeed; }
        public PerformanceConfig setRandomSeed(long s) { this.randomSeed = s; return this; }

        public boolean isVerbose() { return verbose; }
        public PerformanceConfig setVerbose(boolean v) { this.verbose = v; return this; }

        public String getBenchmarkName() { return benchmarkName; }
        public PerformanceConfig setBenchmarkName(String n) { this.benchmarkName = n; return this; }
    }

    /**
     * Builder for PerformanceBenchmark.
     */
    public static class Builder {
        private FaceDetector detector;
        private FeatureExtractor extractor;
        private FaceClassifier classifier;
        private DatasetLoader.LoadedDataset dataset;
        private PerformanceConfig config = new PerformanceConfig();

        public Builder detector(FaceDetector detector) {
            this.detector = detector;
            return this;
        }

        public Builder extractor(FeatureExtractor extractor) {
            this.extractor = extractor;
            return this;
        }

        public Builder classifier(FaceClassifier classifier) {
            this.classifier = classifier;
            return this;
        }

        public Builder dataset(DatasetLoader.LoadedDataset dataset) {
            this.dataset = dataset;
            return this;
        }

        public Builder config(PerformanceConfig config) {
            this.config = config;
            return this;
        }

        public Builder warmupRuns(int warmup) {
            this.config.setWarmupRuns(warmup);
            return this;
        }

        public Builder measurementRuns(int runs) {
            this.config.setMeasurementRuns(runs);
            return this;
        }

        public Builder name(String name) {
            this.config.setBenchmarkName(name);
            return this;
        }

        public PerformanceBenchmark build() {
            Objects.requireNonNull(extractor, "Extractor is required");
            Objects.requireNonNull(classifier, "Classifier is required");
            Objects.requireNonNull(dataset, "Dataset is required");
            return new PerformanceBenchmark(detector, extractor, classifier, dataset, config);
        }
    }

    private PerformanceBenchmark(FaceDetector detector, FeatureExtractor extractor,
                                 FaceClassifier classifier, DatasetLoader.LoadedDataset dataset,
                                 PerformanceConfig config) {
        this.detector = detector;
        this.extractor = extractor;
        this.classifier = classifier;
        this.dataset = dataset;
        this.config = config;
        this.memoryBean = ManagementFactory.getMemoryMXBean();
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
     * Runs the complete performance benchmark.
     *
     * @return benchmark result with performance metrics
     */
    public BenchmarkResult run() {
        if (config.isVerbose()) {
            System.out.printf("Starting performance benchmark: %s%n", config.getBenchmarkName());
            System.out.printf("  Warmup runs: %d, Measurement runs: %d%n",
                config.getWarmupRuns(), config.getMeasurementRuns());
        }

        // Prepare data
        DatasetLoader.DatasetSplit split = dataset.split(config.getTrainRatio(), config.getRandomSeed());
        prepareTraining(split);

        // Force GC before measurements
        if (config.isForceGc()) {
            forceGarbageCollection();
        }

        // Run measurements
        List<Double> detectionTimes = new ArrayList<>();
        List<Double> extractionTimes = new ArrayList<>();
        List<Double> matchingTimes = new ArrayList<>();
        List<Double> totalTimes = new ArrayList<>();
        List<Long> memorySnapshots = new ArrayList<>();

        long peakMemory = 0;
        long baselineMemory = getCurrentMemoryUsage();

        // Warmup phase
        if (config.isVerbose()) {
            System.out.println("  Warming up...");
        }
        runMeasurements(split.getTestSet(), config.getWarmupRuns(),
            null, null, null, null, null);

        // Measurement phase
        if (config.isVerbose()) {
            System.out.println("  Running measurements...");
        }
        long startTime = System.currentTimeMillis();

        runMeasurements(split.getTestSet(), config.getMeasurementRuns(),
            detectionTimes, extractionTimes, matchingTimes, totalTimes, memorySnapshots);

        long endTime = System.currentTimeMillis();

        // Calculate metrics
        BenchmarkResult.TimingStats detectionStats =
            config.isMeasureDetection() && !detectionTimes.isEmpty() ?
                BenchmarkResult.TimingStats.fromValues(detectionTimes) : null;

        BenchmarkResult.TimingStats extractionStats =
            config.isMeasureExtraction() ?
                BenchmarkResult.TimingStats.fromValues(extractionTimes) : null;

        BenchmarkResult.TimingStats matchingStats =
            config.isMeasureMatching() ?
                BenchmarkResult.TimingStats.fromValues(matchingTimes) : null;

        BenchmarkResult.TimingStats totalStats =
            BenchmarkResult.TimingStats.fromValues(totalTimes);

        // Memory statistics
        if (config.isMeasureMemory() && !memorySnapshots.isEmpty()) {
            peakMemory = memorySnapshots.stream().mapToLong(Long::longValue).max().orElse(0);
        }
        long avgMemory = memorySnapshots.isEmpty() ? 0 :
            (long) memorySnapshots.stream().mapToLong(Long::longValue).average().orElse(0);

        // Throughput
        int totalSamples = config.getMeasurementRuns() * split.getTestSize();
        double totalSeconds = (endTime - startTime) / 1000.0;
        double throughput = totalSeconds > 0 ? totalSamples / totalSeconds : 0;

        // Build performance metrics
        BenchmarkResult.PerformanceMetrics performanceMetrics = new BenchmarkResult.PerformanceMetrics(
            detectionStats, extractionStats, matchingStats, totalStats,
            peakMemory - baselineMemory, avgMemory - baselineMemory,
            totalSamples, throughput);

        if (config.isVerbose()) {
            System.out.printf("  Completed. Throughput: %.2f samples/sec%n", throughput);
        }

        return BenchmarkResult.builder()
            .name(config.getBenchmarkName())
            .algorithmName(extractor.getAlgorithmName())
            .description(String.format("Performance benchmark on %s", dataset.getName()))
            .performanceMetrics(performanceMetrics)
            .datasetInfo(dataset.toDatasetInfo(split.getTrainSize(), split.getTestSize()))
            .addConfiguration("warmupRuns", String.valueOf(config.getWarmupRuns()))
            .addConfiguration("measurementRuns", String.valueOf(config.getMeasurementRuns()))
            .build();
    }

    /**
     * Measures detection time only.
     *
     * @param samples list of face images to test
     * @return timing statistics
     */
    public BenchmarkResult.TimingStats measureDetectionTime(List<FaceImage> samples) {
        if (detector == null) {
            throw new IllegalStateException("Detector not configured");
        }

        List<Double> times = new ArrayList<>();

        // Warmup
        for (int i = 0; i < config.getWarmupRuns() && i < samples.size(); i++) {
            detector.detectFaces(samples.get(i));
        }

        // Measure
        for (int run = 0; run < config.getMeasurementRuns(); run++) {
            for (FaceImage sample : samples) {
                long start = System.nanoTime();
                detector.detectFaces(sample);
                long end = System.nanoTime();
                times.add((end - start) / 1_000_000.0);
            }
        }

        return BenchmarkResult.TimingStats.fromValues(times);
    }

    /**
     * Measures feature extraction time only.
     *
     * @param samples list of face images to test
     * @return timing statistics
     */
    public BenchmarkResult.TimingStats measureExtractionTime(List<FaceImage> samples) {
        if (!extractor.isTrained()) {
            throw new IllegalStateException("Extractor not trained");
        }

        List<Double> times = new ArrayList<>();

        // Warmup
        for (int i = 0; i < config.getWarmupRuns() && i < samples.size(); i++) {
            extractor.extract(samples.get(i));
        }

        // Measure
        for (int run = 0; run < config.getMeasurementRuns(); run++) {
            for (FaceImage sample : samples) {
                long start = System.nanoTime();
                extractor.extract(sample);
                long end = System.nanoTime();
                times.add((end - start) / 1_000_000.0);
            }
        }

        return BenchmarkResult.TimingStats.fromValues(times);
    }

    /**
     * Measures matching/classification time only.
     *
     * @param features list of feature vectors to match
     * @return timing statistics
     */
    public BenchmarkResult.TimingStats measureMatchingTime(List<FeatureVector> features) {
        if (classifier.getEnrolledCount() == 0) {
            throw new IllegalStateException("No identities enrolled in classifier");
        }

        List<Double> times = new ArrayList<>();

        // Warmup
        for (int i = 0; i < config.getWarmupRuns() && i < features.size(); i++) {
            classifier.classify(features.get(i));
        }

        // Measure
        for (int run = 0; run < config.getMeasurementRuns(); run++) {
            for (FeatureVector feature : features) {
                long start = System.nanoTime();
                classifier.classify(feature);
                long end = System.nanoTime();
                times.add((end - start) / 1_000_000.0);
            }
        }

        return BenchmarkResult.TimingStats.fromValues(times);
    }

    /**
     * Measures how performance scales with gallery size.
     *
     * @param gallerySizes list of gallery sizes to test
     * @return map of gallery size to timing statistics
     */
    public Map<Integer, BenchmarkResult.TimingStats> measureScalability(List<Integer> gallerySizes) {
        if (config.isVerbose()) {
            System.out.println("  Measuring scalability...");
        }

        DatasetLoader.DatasetSplit split = dataset.split(config.getTrainRatio(), config.getRandomSeed());
        Map<Integer, BenchmarkResult.TimingStats> results = new LinkedHashMap<>();

        // Prepare test features
        List<FaceImage> trainImages = split.getTrainImages();
        List<String> trainLabels = split.getTrainLabels();

        // Train extractor on all data first
        extractor.train(trainImages, trainLabels);

        List<FeatureVector> testFeatures = new ArrayList<>();
        for (DatasetLoader.LabeledFace face : split.getTestSet()) {
            testFeatures.add(extractor.extract(face.getImage()));
        }

        // Test at each gallery size
        for (int gallerySize : gallerySizes) {
            if (config.isVerbose()) {
                System.out.printf("    Testing gallery size: %d%n", gallerySize);
            }

            classifier.clear();

            // Enroll subset of identities
            Map<String, Identity> identities = new LinkedHashMap<>();
            int enrolled = 0;

            for (int i = 0; i < trainImages.size() && enrolled < gallerySize; i++) {
                String label = trainLabels.get(i);
                if (!identities.containsKey(label) && identities.size() < gallerySize) {
                    Identity identity = new Identity(label);
                    FeatureVector features = extractor.extract(trainImages.get(i));
                    identity.enrollSample(features, 1.0, "training");
                    identities.put(label, identity);
                    enrolled++;
                }
            }

            for (Identity identity : identities.values()) {
                classifier.enroll(identity);
            }

            // Measure matching time
            List<Double> times = new ArrayList<>();

            // Warmup
            for (int i = 0; i < config.getWarmupRuns() && i < testFeatures.size(); i++) {
                classifier.classify(testFeatures.get(i));
            }

            // Measure
            for (int run = 0; run < Math.min(config.getMeasurementRuns(), 10); run++) {
                for (FeatureVector feature : testFeatures) {
                    long start = System.nanoTime();
                    classifier.classify(feature);
                    long end = System.nanoTime();
                    times.add((end - start) / 1_000_000.0);
                }
            }

            results.put(gallerySize, BenchmarkResult.TimingStats.fromValues(times));
        }

        return results;
    }

    /**
     * Measures memory usage during operations.
     *
     * @return map of operation name to memory usage in bytes
     */
    public Map<String, Long> measureMemoryUsage() {
        Map<String, Long> usage = new LinkedHashMap<>();

        forceGarbageCollection();
        long baseline = getCurrentMemoryUsage();

        // Train extractor and measure
        DatasetLoader.DatasetSplit split = dataset.split(config.getTrainRatio(), config.getRandomSeed());
        List<FaceImage> trainImages = split.getTrainImages();
        List<String> trainLabels = split.getTrainLabels();

        extractor.reset();
        forceGarbageCollection();
        long beforeTrain = getCurrentMemoryUsage();

        extractor.train(trainImages, trainLabels);
        forceGarbageCollection();
        long afterTrain = getCurrentMemoryUsage();

        usage.put("extractor_training", afterTrain - beforeTrain);

        // Enroll in classifier and measure
        classifier.clear();
        forceGarbageCollection();
        long beforeEnroll = getCurrentMemoryUsage();

        Map<String, Identity> identities = new HashMap<>();
        for (int i = 0; i < trainImages.size(); i++) {
            String label = trainLabels.get(i);
            Identity identity = identities.computeIfAbsent(label, Identity::new);
            FeatureVector features = extractor.extract(trainImages.get(i));
            identity.enrollSample(features, 1.0, "training");
        }
        for (Identity identity : identities.values()) {
            classifier.enroll(identity);
        }

        forceGarbageCollection();
        long afterEnroll = getCurrentMemoryUsage();

        usage.put("classifier_enrollment", afterEnroll - beforeEnroll);

        // Measure per-operation memory
        List<Long> extractionMemory = new ArrayList<>();
        List<Long> matchingMemory = new ArrayList<>();

        for (DatasetLoader.LabeledFace face : split.getTestSet()) {
            long before = getCurrentMemoryUsage();
            FeatureVector features = extractor.extract(face.getImage());
            long afterExtract = getCurrentMemoryUsage();
            extractionMemory.add(afterExtract - before);

            before = getCurrentMemoryUsage();
            classifier.classify(features);
            long afterMatch = getCurrentMemoryUsage();
            matchingMemory.add(afterMatch - before);
        }

        usage.put("extraction_avg", (long) extractionMemory.stream()
            .mapToLong(Long::longValue).average().orElse(0));
        usage.put("matching_avg", (long) matchingMemory.stream()
            .mapToLong(Long::longValue).average().orElse(0));

        return usage;
    }

    /**
     * Compares performance of multiple algorithms.
     *
     * @param algorithms map of algorithm name to extractor supplier
     * @param classifierSupplier classifier supplier
     * @param dataset dataset to use
     * @param config benchmark configuration
     * @return map of algorithm name to benchmark result
     */
    public static Map<String, BenchmarkResult> compareAlgorithms(
            Map<String, Supplier<FeatureExtractor>> algorithms,
            Supplier<FaceClassifier> classifierSupplier,
            DatasetLoader.LoadedDataset dataset,
            PerformanceConfig config) {

        Map<String, BenchmarkResult> results = new LinkedHashMap<>();

        for (Map.Entry<String, Supplier<FeatureExtractor>> entry : algorithms.entrySet()) {
            String algName = entry.getKey();
            System.out.printf("%nBenchmarking %s...%n", algName);

            PerformanceBenchmark benchmark = new PerformanceBenchmark.Builder()
                .extractor(entry.getValue().get())
                .classifier(classifierSupplier.get())
                .dataset(dataset)
                .config(config)
                .name(algName + " Performance")
                .build();

            BenchmarkResult result = benchmark.run();
            results.put(algName, result);

            result.getPerformanceMetrics().ifPresent(metrics ->
                System.out.printf("  Throughput: %.2f/s, Avg time: %.2fms%n",
                    metrics.getThroughput(),
                    metrics.getTotalTime() != null ? metrics.getTotalTime().getMean() : 0));
        }

        return results;
    }

    /**
     * Generates a summary comparison of algorithm performance.
     *
     * @param results map of algorithm name to benchmark result
     * @return formatted comparison string
     */
    public static String generateComparisonSummary(Map<String, BenchmarkResult> results) {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Performance Comparison ===\n\n");
        sb.append(String.format("%-20s %12s %12s %12s %12s%n",
            "Algorithm", "Throughput", "Mean (ms)", "P95 (ms)", "Memory (MB)"));
        sb.append("-".repeat(72)).append("\n");

        for (Map.Entry<String, BenchmarkResult> entry : results.entrySet()) {
            String name = entry.getKey();
            BenchmarkResult result = entry.getValue();

            result.getPerformanceMetrics().ifPresent(metrics -> {
                BenchmarkResult.TimingStats totalStats = metrics.getTotalTime();
                sb.append(String.format("%-20s %12.2f %12.2f %12.2f %12.2f%n",
                    truncate(name, 20),
                    metrics.getThroughput(),
                    totalStats != null ? totalStats.getMean() : 0,
                    totalStats != null ? totalStats.getP95() : 0,
                    metrics.getPeakMemoryMB()));
            });
        }

        return sb.toString();
    }

    private static String truncate(String s, int maxLen) {
        return s.length() <= maxLen ? s : s.substring(0, maxLen - 2) + "..";
    }

    /**
     * Prepares training data.
     */
    private void prepareTraining(DatasetLoader.DatasetSplit split) {
        List<FaceImage> trainImages = split.getTrainImages();
        List<String> trainLabels = split.getTrainLabels();

        // Train extractor
        extractor.train(trainImages, trainLabels);

        // Enroll identities
        classifier.clear();
        Map<String, Identity> identities = new HashMap<>();

        for (int i = 0; i < trainImages.size(); i++) {
            String label = trainLabels.get(i);
            Identity identity = identities.computeIfAbsent(label, Identity::new);
            FeatureVector features = extractor.extract(trainImages.get(i));
            identity.enrollSample(features, 1.0, "training");
        }

        for (Identity identity : identities.values()) {
            classifier.enroll(identity);
        }
    }

    /**
     * Runs measurement iterations.
     */
    private void runMeasurements(List<DatasetLoader.LabeledFace> testSet, int runs,
                                 List<Double> detectionTimes,
                                 List<Double> extractionTimes,
                                 List<Double> matchingTimes,
                                 List<Double> totalTimes,
                                 List<Long> memorySnapshots) {

        for (int run = 0; run < runs; run++) {
            for (DatasetLoader.LabeledFace face : testSet) {
                long totalStart = System.nanoTime();
                long detectionTime = 0, extractionTime = 0, matchingTime = 0;

                // Detection (if detector available)
                if (config.isMeasureDetection() && detector != null) {
                    long start = System.nanoTime();
                    detector.detectFaces(face.getImage());
                    detectionTime = System.nanoTime() - start;
                }

                // Extraction
                long extractStart = System.nanoTime();
                FeatureVector features = extractor.extract(face.getImage());
                extractionTime = System.nanoTime() - extractStart;

                // Matching
                long matchStart = System.nanoTime();
                classifier.classify(features);
                matchingTime = System.nanoTime() - matchStart;

                long totalTime = System.nanoTime() - totalStart;

                // Record measurements
                if (detectionTimes != null && detectionTime > 0) {
                    detectionTimes.add(detectionTime / 1_000_000.0);
                }
                if (extractionTimes != null) {
                    extractionTimes.add(extractionTime / 1_000_000.0);
                }
                if (matchingTimes != null) {
                    matchingTimes.add(matchingTime / 1_000_000.0);
                }
                if (totalTimes != null) {
                    totalTimes.add(totalTime / 1_000_000.0);
                }

                // Memory snapshot
                if (memorySnapshots != null && config.isMeasureMemory()) {
                    memorySnapshots.add(getCurrentMemoryUsage());
                }
            }
        }
    }

    /**
     * Gets current memory usage in bytes.
     */
    private long getCurrentMemoryUsage() {
        MemoryUsage heapUsage = memoryBean.getHeapMemoryUsage();
        return heapUsage.getUsed();
    }

    /**
     * Forces garbage collection for more accurate memory measurements.
     */
    private void forceGarbageCollection() {
        System.gc();
        System.gc();
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
}
