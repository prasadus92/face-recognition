package com.facerecognition.benchmark;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.domain.service.FaceClassifier;

import java.util.*;
import java.util.function.Supplier;

/**
 * Accuracy benchmarking for face recognition algorithms.
 *
 * <p>This class provides comprehensive accuracy testing capabilities:</p>
 * <ul>
 *   <li><b>Recognition Accuracy</b>: Measures overall classification performance</li>
 *   <li><b>Verification Accuracy</b>: Measures 1:1 matching performance</li>
 *   <li><b>Algorithm Comparison</b>: Compares multiple algorithms on same data</li>
 *   <li><b>Condition Testing</b>: Tests under various conditions (pose, lighting, etc.)</li>
 *   <li><b>Threshold Analysis</b>: Evaluates performance across different thresholds</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * AccuracyBenchmark benchmark = new AccuracyBenchmark.Builder()
 *     .extractor(new EigenfacesExtractor(10))
 *     .classifier(new KNNClassifier())
 *     .dataset(dataset)
 *     .trainRatio(0.7)
 *     .build();
 *
 * BenchmarkResult result = benchmark.run();
 * System.out.println("Accuracy: " + result.getAccuracy());
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see BenchmarkResult
 * @see CrossValidator
 */
public class AccuracyBenchmark {

    private final FeatureExtractor extractor;
    private final FaceClassifier classifier;
    private final DatasetLoader.LoadedDataset dataset;
    private final BenchmarkConfig config;

    /**
     * Configuration for accuracy benchmarks.
     */
    public static class BenchmarkConfig {
        private double trainRatio = 0.7;
        private double recognitionThreshold = 0.6;
        private boolean computeRocCurve = true;
        private boolean computeConfusionMatrix = true;
        private boolean computePerClassMetrics = true;
        private int rocCurvePoints = 100;
        private long randomSeed = 42;
        private boolean verbose = true;
        private String benchmarkName = "Accuracy Benchmark";

        public double getTrainRatio() { return trainRatio; }
        public BenchmarkConfig setTrainRatio(double r) { this.trainRatio = r; return this; }

        public double getRecognitionThreshold() { return recognitionThreshold; }
        public BenchmarkConfig setRecognitionThreshold(double t) { this.recognitionThreshold = t; return this; }

        public boolean isComputeRocCurve() { return computeRocCurve; }
        public BenchmarkConfig setComputeRocCurve(boolean c) { this.computeRocCurve = c; return this; }

        public boolean isComputeConfusionMatrix() { return computeConfusionMatrix; }
        public BenchmarkConfig setComputeConfusionMatrix(boolean c) { this.computeConfusionMatrix = c; return this; }

        public boolean isComputePerClassMetrics() { return computePerClassMetrics; }
        public BenchmarkConfig setComputePerClassMetrics(boolean c) { this.computePerClassMetrics = c; return this; }

        public int getRocCurvePoints() { return rocCurvePoints; }
        public BenchmarkConfig setRocCurvePoints(int p) { this.rocCurvePoints = p; return this; }

        public long getRandomSeed() { return randomSeed; }
        public BenchmarkConfig setRandomSeed(long s) { this.randomSeed = s; return this; }

        public boolean isVerbose() { return verbose; }
        public BenchmarkConfig setVerbose(boolean v) { this.verbose = v; return this; }

        public String getBenchmarkName() { return benchmarkName; }
        public BenchmarkConfig setBenchmarkName(String n) { this.benchmarkName = n; return this; }
    }

    /**
     * Stores a test prediction with detailed information.
     */
    public static class PredictionRecord {
        private final String actualLabel;
        private final String predictedLabel;
        private final double confidence;
        private final double distance;
        private final FeatureVector features;

        public PredictionRecord(String actualLabel, String predictedLabel,
                               double confidence, double distance, FeatureVector features) {
            this.actualLabel = actualLabel;
            this.predictedLabel = predictedLabel;
            this.confidence = confidence;
            this.distance = distance;
            this.features = features;
        }

        public String getActualLabel() { return actualLabel; }
        public String getPredictedLabel() { return predictedLabel; }
        public double getConfidence() { return confidence; }
        public double getDistance() { return distance; }
        public FeatureVector getFeatures() { return features; }
        public boolean isCorrect() { return actualLabel.equals(predictedLabel); }

        @Override
        public String toString() {
            return String.format("Prediction{actual='%s', predicted='%s', conf=%.4f}",
                actualLabel, predictedLabel, confidence);
        }
    }

    /**
     * Stores a verification pair test result.
     */
    public static class VerificationTestResult {
        private final boolean samePerson;
        private final boolean predictedSame;
        private final double similarity;
        private final double distance;
        private final String person1;
        private final String person2;

        public VerificationTestResult(boolean samePerson, boolean predictedSame,
                                     double similarity, double distance,
                                     String person1, String person2) {
            this.samePerson = samePerson;
            this.predictedSame = predictedSame;
            this.similarity = similarity;
            this.distance = distance;
            this.person1 = person1;
            this.person2 = person2;
        }

        public boolean isSamePerson() { return samePerson; }
        public boolean isPredictedSame() { return predictedSame; }
        public double getSimilarity() { return similarity; }
        public double getDistance() { return distance; }
        public String getPerson1() { return person1; }
        public String getPerson2() { return person2; }
        public boolean isCorrect() { return samePerson == predictedSame; }
        public boolean isTruePositive() { return samePerson && predictedSame; }
        public boolean isFalsePositive() { return !samePerson && predictedSame; }
        public boolean isTrueNegative() { return !samePerson && !predictedSame; }
        public boolean isFalseNegative() { return samePerson && !predictedSame; }
    }

    /**
     * Builder for AccuracyBenchmark.
     */
    public static class Builder {
        private FeatureExtractor extractor;
        private FaceClassifier classifier;
        private DatasetLoader.LoadedDataset dataset;
        private BenchmarkConfig config = new BenchmarkConfig();

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

        public Builder config(BenchmarkConfig config) {
            this.config = config;
            return this;
        }

        public Builder trainRatio(double ratio) {
            this.config.setTrainRatio(ratio);
            return this;
        }

        public Builder threshold(double threshold) {
            this.config.setRecognitionThreshold(threshold);
            return this;
        }

        public Builder seed(long seed) {
            this.config.setRandomSeed(seed);
            return this;
        }

        public Builder name(String name) {
            this.config.setBenchmarkName(name);
            return this;
        }

        public AccuracyBenchmark build() {
            Objects.requireNonNull(extractor, "Extractor is required");
            Objects.requireNonNull(classifier, "Classifier is required");
            Objects.requireNonNull(dataset, "Dataset is required");
            return new AccuracyBenchmark(extractor, classifier, dataset, config);
        }
    }

    private AccuracyBenchmark(FeatureExtractor extractor, FaceClassifier classifier,
                              DatasetLoader.LoadedDataset dataset, BenchmarkConfig config) {
        this.extractor = extractor;
        this.classifier = classifier;
        this.dataset = dataset;
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
     * Runs the accuracy benchmark.
     *
     * @return benchmark result with all metrics
     */
    public BenchmarkResult run() {
        if (config.isVerbose()) {
            System.out.printf("Starting accuracy benchmark: %s%n", config.getBenchmarkName());
            System.out.printf("  Dataset: %s (%d classes, %d images)%n",
                dataset.getName(), dataset.getNumClasses(), dataset.getFaces().size());
            System.out.printf("  Train ratio: %.0f%%  Threshold: %.2f%n",
                config.getTrainRatio() * 100, config.getRecognitionThreshold());
        }

        // Split dataset
        DatasetLoader.DatasetSplit split = dataset.split(config.getTrainRatio(), config.getRandomSeed());

        // Train
        if (config.isVerbose()) {
            System.out.println("  Training...");
        }
        trainOnSplit(split);

        // Test
        if (config.isVerbose()) {
            System.out.println("  Testing...");
        }
        List<PredictionRecord> predictions = testOnSplit(split);

        // Compute metrics
        BenchmarkResult.Builder resultBuilder = BenchmarkResult.builder()
            .name(config.getBenchmarkName())
            .algorithmName(extractor.getAlgorithmName())
            .description(String.format("Recognition accuracy on %s", dataset.getName()))
            .datasetInfo(dataset.toDatasetInfo(split.getTrainSize(), split.getTestSize()));

        // Add configuration
        resultBuilder.addConfiguration("trainRatio", String.valueOf(config.getTrainRatio()));
        resultBuilder.addConfiguration("threshold", String.valueOf(config.getRecognitionThreshold()));
        resultBuilder.addConfiguration("seed", String.valueOf(config.getRandomSeed()));

        // Compute basic metrics
        computeBasicMetrics(predictions, resultBuilder);

        // Confusion matrix
        if (config.isComputeConfusionMatrix()) {
            BenchmarkResult.ConfusionMatrix cm = computeConfusionMatrix(predictions);
            resultBuilder.confusionMatrix(cm);
            resultBuilder.computeMetricsFromConfusionMatrix();
        }

        // ROC curve and error rates
        if (config.isComputeRocCurve()) {
            computeRocAndErrorRates(predictions, resultBuilder);
        }

        if (config.isVerbose()) {
            System.out.println("  Benchmark complete.");
        }

        return resultBuilder.build();
    }

    /**
     * Runs verification benchmark on pair data.
     *
     * @param pairs list of verification pairs
     * @return benchmark result
     */
    public BenchmarkResult runVerification(List<DatasetLoader.VerificationPair> pairs) {
        if (config.isVerbose()) {
            System.out.printf("Starting verification benchmark: %s%n", config.getBenchmarkName());
            System.out.printf("  Pairs: %d%n", pairs.size());
        }

        // Train on all unique faces
        trainOnPairs(pairs);

        // Test each pair
        List<VerificationTestResult> results = new ArrayList<>();
        for (DatasetLoader.VerificationPair pair : pairs) {
            VerificationTestResult result = testVerificationPair(pair);
            results.add(result);
        }

        // Compute metrics
        return computeVerificationMetrics(results);
    }

    /**
     * Compares multiple algorithms on the same dataset.
     *
     * @param algorithms map of algorithm name to extractor supplier
     * @param classifierSupplier classifier supplier
     * @param dataset dataset to evaluate
     * @param config benchmark configuration
     * @return map of algorithm name to result
     */
    public static Map<String, BenchmarkResult> compareAlgorithms(
            Map<String, Supplier<FeatureExtractor>> algorithms,
            Supplier<FaceClassifier> classifierSupplier,
            DatasetLoader.LoadedDataset dataset,
            BenchmarkConfig config) {

        Map<String, BenchmarkResult> results = new LinkedHashMap<>();

        for (Map.Entry<String, Supplier<FeatureExtractor>> entry : algorithms.entrySet()) {
            String algName = entry.getKey();
            System.out.printf("%nEvaluating %s...%n", algName);

            AccuracyBenchmark benchmark = new AccuracyBenchmark.Builder()
                .extractor(entry.getValue().get())
                .classifier(classifierSupplier.get())
                .dataset(dataset)
                .config(config)
                .name(algName + " on " + dataset.getName())
                .build();

            BenchmarkResult result = benchmark.run();
            results.put(algName, result);

            System.out.printf("  Accuracy: %.4f, F1: %.4f%n",
                result.getAccuracy(), result.getF1Score());
        }

        return results;
    }

    /**
     * Tests accuracy under different conditions.
     *
     * @param conditions map of condition name to faces with that condition
     * @return map of condition name to benchmark result
     */
    public Map<String, BenchmarkResult> testUnderConditions(
            Map<String, List<DatasetLoader.LabeledFace>> conditions) {

        Map<String, BenchmarkResult> results = new LinkedHashMap<>();

        // First, train on full training set
        DatasetLoader.DatasetSplit baseSplit = dataset.split(config.getTrainRatio(), config.getRandomSeed());
        trainOnSplit(baseSplit);

        // Test under each condition
        for (Map.Entry<String, List<DatasetLoader.LabeledFace>> entry : conditions.entrySet()) {
            String conditionName = entry.getKey();
            List<DatasetLoader.LabeledFace> conditionFaces = entry.getValue();

            if (config.isVerbose()) {
                System.out.printf("  Testing condition: %s (%d samples)%n",
                    conditionName, conditionFaces.size());
            }

            // Test these faces
            List<PredictionRecord> predictions = new ArrayList<>();
            for (DatasetLoader.LabeledFace face : conditionFaces) {
                FeatureVector features = extractor.extract(face.getImage());
                RecognitionResult result = classifier.classify(features, config.getRecognitionThreshold());

                String predictedLabel = result.getIdentity()
                    .map(Identity::getName)
                    .orElse("UNKNOWN");

                predictions.add(new PredictionRecord(
                    face.getLabel(), predictedLabel,
                    result.getConfidence(), result.getDistance(), features));
            }

            // Compute metrics
            BenchmarkResult.Builder resultBuilder = BenchmarkResult.builder()
                .name(config.getBenchmarkName() + " - " + conditionName)
                .algorithmName(extractor.getAlgorithmName())
                .description("Accuracy under " + conditionName + " condition");

            computeBasicMetrics(predictions, resultBuilder);
            results.put(conditionName, resultBuilder.build());
        }

        return results;
    }

    /**
     * Evaluates performance across different threshold values.
     *
     * @param thresholds list of threshold values to test
     * @return map of threshold to accuracy
     */
    public Map<Double, Double> thresholdAnalysis(List<Double> thresholds) {
        // Split and train
        DatasetLoader.DatasetSplit split = dataset.split(config.getTrainRatio(), config.getRandomSeed());
        trainOnSplit(split);

        // Extract features for all test samples
        List<TestSample> testSamples = new ArrayList<>();
        for (DatasetLoader.LabeledFace face : split.getTestSet()) {
            FeatureVector features = extractor.extract(face.getImage());
            testSamples.add(new TestSample(face.getLabel(), features));
        }

        // Test at each threshold
        Map<Double, Double> results = new LinkedHashMap<>();
        for (double threshold : thresholds) {
            int correct = 0;
            for (TestSample sample : testSamples) {
                RecognitionResult result = classifier.classify(sample.features, threshold);
                String predicted = result.getIdentity()
                    .map(Identity::getName)
                    .orElse("UNKNOWN");

                if (sample.label.equals(predicted)) {
                    correct++;
                }
            }
            double accuracy = (double) correct / testSamples.size();
            results.put(threshold, accuracy);
        }

        return results;
    }

    /**
     * Evaluates performance with varying number of training samples.
     *
     * @param trainRatios list of training ratios to test
     * @return map of train ratio to benchmark result
     */
    public Map<Double, BenchmarkResult> learningCurve(List<Double> trainRatios) {
        Map<Double, BenchmarkResult> results = new LinkedHashMap<>();

        for (double ratio : trainRatios) {
            if (config.isVerbose()) {
                System.out.printf("  Testing train ratio: %.0f%%...%n", ratio * 100);
            }

            // Create fresh instances
            extractor.reset();
            classifier.clear();

            // Set train ratio and run
            BenchmarkConfig tempConfig = new BenchmarkConfig();
            tempConfig.setTrainRatio(ratio);
            tempConfig.setRecognitionThreshold(config.getRecognitionThreshold());
            tempConfig.setRandomSeed(config.getRandomSeed());
            tempConfig.setComputeRocCurve(false);
            tempConfig.setVerbose(false);

            AccuracyBenchmark tempBenchmark = new AccuracyBenchmark(
                extractor, classifier, dataset, tempConfig);

            BenchmarkResult result = tempBenchmark.run();
            results.put(ratio, result);
        }

        return results;
    }

    // Helper class for test samples
    private static class TestSample {
        final String label;
        final FeatureVector features;

        TestSample(String label, FeatureVector features) {
            this.label = label;
            this.features = features;
        }
    }

    /**
     * Trains the extractor and classifier on the training split.
     */
    private void trainOnSplit(DatasetLoader.DatasetSplit split) {
        List<FaceImage> trainImages = split.getTrainImages();
        List<String> trainLabels = split.getTrainLabels();

        // Train extractor
        extractor.train(trainImages, trainLabels);

        // Extract features and enroll in classifier
        classifier.clear();
        Map<String, Identity> identities = new HashMap<>();

        for (int i = 0; i < trainImages.size(); i++) {
            FeatureVector features = extractor.extract(trainImages.get(i));
            String label = trainLabels.get(i);

            Identity identity = identities.computeIfAbsent(label, Identity::new);
            identity.enrollSample(features, 1.0, "training");
        }

        for (Identity identity : identities.values()) {
            classifier.enroll(identity);
        }
    }

    /**
     * Trains on faces from verification pairs.
     */
    private void trainOnPairs(List<DatasetLoader.VerificationPair> pairs) {
        // Collect all unique faces
        Map<String, List<FaceImage>> facesByPerson = new HashMap<>();

        for (DatasetLoader.VerificationPair pair : pairs) {
            String label1 = pair.getFace1().getLabel();
            String label2 = pair.getFace2().getLabel();

            facesByPerson.computeIfAbsent(label1, k -> new ArrayList<>())
                .add(pair.getFace1().getImage());
            facesByPerson.computeIfAbsent(label2, k -> new ArrayList<>())
                .add(pair.getFace2().getImage());
        }

        // Flatten for training
        List<FaceImage> images = new ArrayList<>();
        List<String> labels = new ArrayList<>();
        for (Map.Entry<String, List<FaceImage>> entry : facesByPerson.entrySet()) {
            for (FaceImage image : entry.getValue()) {
                images.add(image);
                labels.add(entry.getKey());
            }
        }

        // Train extractor
        extractor.train(images, labels);
    }

    /**
     * Tests on the test split and returns prediction records.
     */
    private List<PredictionRecord> testOnSplit(DatasetLoader.DatasetSplit split) {
        List<PredictionRecord> predictions = new ArrayList<>();

        for (DatasetLoader.LabeledFace face : split.getTestSet()) {
            FeatureVector features = extractor.extract(face.getImage());
            RecognitionResult result = classifier.classify(features, config.getRecognitionThreshold());

            String predictedLabel = result.getIdentity()
                .map(Identity::getName)
                .orElse("UNKNOWN");

            predictions.add(new PredictionRecord(
                face.getLabel(), predictedLabel,
                result.getConfidence(), result.getDistance(), features));
        }

        return predictions;
    }

    /**
     * Tests a verification pair.
     */
    private VerificationTestResult testVerificationPair(DatasetLoader.VerificationPair pair) {
        FeatureVector features1 = extractor.extract(pair.getFace1().getImage());
        FeatureVector features2 = extractor.extract(pair.getFace2().getImage());

        double distance = features1.euclideanDistance(features2);
        double similarity = features1.cosineSimilarity(features2);

        // Threshold-based decision
        boolean predictedSame = distance < config.getRecognitionThreshold();

        return new VerificationTestResult(
            pair.isSamePerson(), predictedSame,
            similarity, distance,
            pair.getFace1().getLabel(), pair.getFace2().getLabel());
    }

    /**
     * Computes basic metrics from predictions.
     */
    private void computeBasicMetrics(List<PredictionRecord> predictions,
                                     BenchmarkResult.Builder builder) {
        int correct = 0;
        int total = predictions.size();

        // Per-class counters
        Map<String, int[]> classStats = new HashMap<>(); // [TP, FP, FN]

        for (PredictionRecord pred : predictions) {
            String actual = pred.getActualLabel();
            String predicted = pred.getPredictedLabel();

            classStats.putIfAbsent(actual, new int[3]);

            if (actual.equals(predicted)) {
                correct++;
                classStats.get(actual)[0]++; // TP
            } else {
                classStats.get(actual)[2]++; // FN
                if (!"UNKNOWN".equals(predicted)) {
                    classStats.putIfAbsent(predicted, new int[3]);
                    classStats.get(predicted)[1]++; // FP
                }
            }
        }

        double accuracy = total > 0 ? (double) correct / total : 0.0;
        builder.accuracy(accuracy);

        // Macro-averaged precision, recall, F1
        double totalPrecision = 0, totalRecall = 0;
        int validClasses = 0;

        for (int[] stats : classStats.values()) {
            int tp = stats[0], fp = stats[1], fn = stats[2];
            if (tp + fp > 0 || tp + fn > 0) {
                validClasses++;
                totalPrecision += (tp + fp > 0) ? (double) tp / (tp + fp) : 0;
                totalRecall += (tp + fn > 0) ? (double) tp / (tp + fn) : 0;
            }
        }

        double precision = validClasses > 0 ? totalPrecision / validClasses : 0;
        double recall = validClasses > 0 ? totalRecall / validClasses : 0;
        double f1 = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0;

        builder.precision(precision);
        builder.recall(recall);
        builder.f1Score(f1);
    }

    /**
     * Computes confusion matrix from predictions.
     */
    private BenchmarkResult.ConfusionMatrix computeConfusionMatrix(List<PredictionRecord> predictions) {
        // Collect unique labels
        Set<String> labelSet = new TreeSet<>();
        for (PredictionRecord pred : predictions) {
            labelSet.add(pred.getActualLabel());
            if (!"UNKNOWN".equals(pred.getPredictedLabel())) {
                labelSet.add(pred.getPredictedLabel());
            }
        }

        List<String> labels = new ArrayList<>(labelSet);
        Map<String, Integer> labelIndex = new HashMap<>();
        for (int i = 0; i < labels.size(); i++) {
            labelIndex.put(labels.get(i), i);
        }

        // Build matrix
        int[][] matrix = new int[labels.size()][labels.size()];
        for (PredictionRecord pred : predictions) {
            Integer actualIdx = labelIndex.get(pred.getActualLabel());
            Integer predictedIdx = labelIndex.get(pred.getPredictedLabel());

            if (actualIdx != null && predictedIdx != null) {
                matrix[actualIdx][predictedIdx]++;
            }
        }

        return new BenchmarkResult.ConfusionMatrix(labels, matrix);
    }

    /**
     * Computes ROC curve and error rates.
     */
    private void computeRocAndErrorRates(List<PredictionRecord> predictions,
                                         BenchmarkResult.Builder builder) {
        // Sort by distance (ascending)
        List<PredictionRecord> sorted = new ArrayList<>(predictions);
        sorted.sort(Comparator.comparingDouble(PredictionRecord::getDistance));

        int totalPositives = (int) sorted.stream().filter(PredictionRecord::isCorrect).count();
        int totalNegatives = sorted.size() - totalPositives;

        // Generate ROC points
        List<BenchmarkResult.RocPoint> rocPoints = new ArrayList<>();
        List<BenchmarkResult.DetPoint> detPoints = new ArrayList<>();

        // Add initial points
        rocPoints.add(new BenchmarkResult.RocPoint(0, 0, 0));
        detPoints.add(new BenchmarkResult.DetPoint(0, 1, 0));

        // Sweep through thresholds
        double minDist = sorted.isEmpty() ? 0 : sorted.get(0).getDistance();
        double maxDist = sorted.isEmpty() ? 1 : sorted.get(sorted.size() - 1).getDistance();
        double step = (maxDist - minDist) / config.getRocCurvePoints();

        double eer = 0;
        double minEerDiff = Double.MAX_VALUE;

        for (int i = 0; i <= config.getRocCurvePoints(); i++) {
            double threshold = minDist + i * step;

            int tp = 0, fp = 0, tn = 0, fn = 0;

            for (PredictionRecord pred : sorted) {
                boolean actualPositive = pred.isCorrect();
                boolean predictedPositive = pred.getDistance() <= threshold;

                if (actualPositive && predictedPositive) tp++;
                else if (!actualPositive && predictedPositive) fp++;
                else if (!actualPositive) tn++;
                else fn++;
            }

            double tpr = totalPositives > 0 ? (double) tp / totalPositives : 0;
            double fpr = totalNegatives > 0 ? (double) fp / totalNegatives : 0;
            double frr = totalPositives > 0 ? (double) fn / totalPositives : 0;
            double far = totalNegatives > 0 ? (double) fp / totalNegatives : 0;

            rocPoints.add(new BenchmarkResult.RocPoint(threshold, tpr, fpr));
            detPoints.add(new BenchmarkResult.DetPoint(threshold, frr, far));

            // Find EER (where FAR == FRR)
            double eerDiff = Math.abs(far - frr);
            if (eerDiff < minEerDiff) {
                minEerDiff = eerDiff;
                eer = (far + frr) / 2;
            }
        }

        // Add final points
        rocPoints.add(new BenchmarkResult.RocPoint(maxDist + step, 1, 1));
        detPoints.add(new BenchmarkResult.DetPoint(maxDist + step, 0, 1));

        builder.rocCurve(rocPoints);
        builder.detCurve(detPoints);
        builder.equalErrorRate(eer);

        // Compute AUC using trapezoidal rule
        rocPoints.sort(Comparator.comparingDouble(BenchmarkResult.RocPoint::getFalsePositiveRate));
        double auc = 0;
        for (int i = 1; i < rocPoints.size(); i++) {
            BenchmarkResult.RocPoint p1 = rocPoints.get(i - 1);
            BenchmarkResult.RocPoint p2 = rocPoints.get(i);
            double width = p2.getFalsePositiveRate() - p1.getFalsePositiveRate();
            double avgHeight = (p1.getTruePositiveRate() + p2.getTruePositiveRate()) / 2;
            auc += width * avgHeight;
        }
        builder.areaUnderCurve(auc);

        // Set FAR and FRR at operating threshold
        double opThreshold = config.getRecognitionThreshold();
        for (BenchmarkResult.DetPoint dp : detPoints) {
            if (Math.abs(dp.getThreshold() - opThreshold) < step) {
                builder.falseAcceptRate(dp.getFalseAcceptRate());
                builder.falseRejectRate(dp.getFalseRejectRate());
                break;
            }
        }
    }

    /**
     * Computes verification metrics from test results.
     */
    private BenchmarkResult computeVerificationMetrics(List<VerificationTestResult> results) {
        int tp = 0, fp = 0, tn = 0, fn = 0;

        for (VerificationTestResult r : results) {
            if (r.isTruePositive()) tp++;
            else if (r.isFalsePositive()) fp++;
            else if (r.isTrueNegative()) tn++;
            else fn++;
        }

        double accuracy = results.size() > 0 ? (double) (tp + tn) / results.size() : 0;
        double precision = tp + fp > 0 ? (double) tp / (tp + fp) : 0;
        double recall = tp + fn > 0 ? (double) tp / (tp + fn) : 0;
        double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
        double far = fp + tn > 0 ? (double) fp / (fp + tn) : 0;
        double frr = tp + fn > 0 ? (double) fn / (tp + fn) : 0;

        return BenchmarkResult.builder()
            .name(config.getBenchmarkName())
            .algorithmName(extractor.getAlgorithmName())
            .description("Verification accuracy")
            .accuracy(accuracy)
            .precision(precision)
            .recall(recall)
            .f1Score(f1)
            .falseAcceptRate(far)
            .falseRejectRate(frr)
            .build();
    }
}
