package com.facerecognition.benchmark;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.domain.service.FaceClassifier;

import java.util.*;
import java.util.function.Supplier;
import java.util.stream.Collectors;

/**
 * Cross-validation utilities for evaluating face recognition algorithms.
 *
 * <p>This class provides various cross-validation strategies:</p>
 * <ul>
 *   <li><b>K-Fold Cross Validation</b>: Divides data into K equal folds</li>
 *   <li><b>Stratified K-Fold</b>: Maintains class distribution in each fold</li>
 *   <li><b>Leave-One-Out (LOO)</b>: Each sample is tested individually</li>
 *   <li><b>Leave-P-Out</b>: Leave P samples out per class</li>
 *   <li><b>Repeated K-Fold</b>: Runs K-fold multiple times with different seeds</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * CrossValidator validator = new CrossValidator(extractorSupplier, classifierSupplier);
 *
 * // Perform 5-fold cross validation
 * CrossValidationResult result = validator.kFoldCrossValidation(dataset, 5);
 *
 * System.out.println("Average accuracy: " + result.getMeanAccuracy());
 * System.out.println("Standard deviation: " + result.getStdAccuracy());
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see DatasetLoader
 * @see BenchmarkResult
 */
public class CrossValidator {

    private final Supplier<FeatureExtractor> extractorSupplier;
    private final Supplier<FaceClassifier> classifierSupplier;
    private final CrossValidationConfig config;

    /**
     * Configuration for cross-validation.
     */
    public static class CrossValidationConfig {
        private boolean verbose = true;
        private boolean collectPerFoldResults = true;
        private double recognitionThreshold = 0.6;
        private int targetWidth = 48;
        private int targetHeight = 64;

        public boolean isVerbose() { return verbose; }
        public CrossValidationConfig setVerbose(boolean v) { this.verbose = v; return this; }

        public boolean isCollectPerFoldResults() { return collectPerFoldResults; }
        public CrossValidationConfig setCollectPerFoldResults(boolean c) { this.collectPerFoldResults = c; return this; }

        public double getRecognitionThreshold() { return recognitionThreshold; }
        public CrossValidationConfig setRecognitionThreshold(double t) { this.recognitionThreshold = t; return this; }

        public int getTargetWidth() { return targetWidth; }
        public CrossValidationConfig setTargetWidth(int w) { this.targetWidth = w; return this; }

        public int getTargetHeight() { return targetHeight; }
        public CrossValidationConfig setTargetHeight(int h) { this.targetHeight = h; return this; }
    }

    /**
     * Result of a cross-validation run.
     */
    public static class CrossValidationResult {
        private final List<FoldResult> foldResults;
        private final String strategyName;
        private final int numFolds;
        private final long totalTimeMs;

        public CrossValidationResult(String strategyName, List<FoldResult> foldResults, long totalTimeMs) {
            this.strategyName = strategyName;
            this.foldResults = new ArrayList<>(foldResults);
            this.numFolds = foldResults.size();
            this.totalTimeMs = totalTimeMs;
        }

        public List<FoldResult> getFoldResults() { return Collections.unmodifiableList(foldResults); }
        public String getStrategyName() { return strategyName; }
        public int getNumFolds() { return numFolds; }
        public long getTotalTimeMs() { return totalTimeMs; }

        /**
         * Gets the mean accuracy across all folds.
         *
         * @return mean accuracy (0.0 to 1.0)
         */
        public double getMeanAccuracy() {
            return foldResults.stream()
                .mapToDouble(FoldResult::getAccuracy)
                .average()
                .orElse(0.0);
        }

        /**
         * Gets the standard deviation of accuracy across folds.
         *
         * @return standard deviation
         */
        public double getStdAccuracy() {
            double mean = getMeanAccuracy();
            double variance = foldResults.stream()
                .mapToDouble(r -> Math.pow(r.getAccuracy() - mean, 2))
                .average()
                .orElse(0.0);
            return Math.sqrt(variance);
        }

        /**
         * Gets the mean precision across all folds.
         *
         * @return mean precision
         */
        public double getMeanPrecision() {
            return foldResults.stream()
                .mapToDouble(FoldResult::getPrecision)
                .average()
                .orElse(0.0);
        }

        /**
         * Gets the standard deviation of precision.
         *
         * @return std precision
         */
        public double getStdPrecision() {
            double mean = getMeanPrecision();
            return Math.sqrt(foldResults.stream()
                .mapToDouble(r -> Math.pow(r.getPrecision() - mean, 2))
                .average()
                .orElse(0.0));
        }

        /**
         * Gets the mean recall across all folds.
         *
         * @return mean recall
         */
        public double getMeanRecall() {
            return foldResults.stream()
                .mapToDouble(FoldResult::getRecall)
                .average()
                .orElse(0.0);
        }

        /**
         * Gets the standard deviation of recall.
         *
         * @return std recall
         */
        public double getStdRecall() {
            double mean = getMeanRecall();
            return Math.sqrt(foldResults.stream()
                .mapToDouble(r -> Math.pow(r.getRecall() - mean, 2))
                .average()
                .orElse(0.0));
        }

        /**
         * Gets the mean F1 score across all folds.
         *
         * @return mean F1
         */
        public double getMeanF1() {
            return foldResults.stream()
                .mapToDouble(FoldResult::getF1Score)
                .average()
                .orElse(0.0);
        }

        /**
         * Gets the standard deviation of F1 score.
         *
         * @return std F1
         */
        public double getStdF1() {
            double mean = getMeanF1();
            return Math.sqrt(foldResults.stream()
                .mapToDouble(r -> Math.pow(r.getF1Score() - mean, 2))
                .average()
                .orElse(0.0));
        }

        /**
         * Gets the confidence interval for accuracy at 95% confidence level.
         *
         * @return array of [lower, upper] bounds
         */
        public double[] getAccuracyConfidenceInterval() {
            double mean = getMeanAccuracy();
            double std = getStdAccuracy();
            double se = std / Math.sqrt(numFolds);
            double margin = 1.96 * se; // 95% CI

            return new double[]{
                Math.max(0, mean - margin),
                Math.min(1, mean + margin)
            };
        }

        /**
         * Converts this result to a BenchmarkResult.
         *
         * @param name the benchmark name
         * @param algorithmName the algorithm name
         * @return aggregated BenchmarkResult
         */
        public BenchmarkResult toBenchmarkResult(String name, String algorithmName) {
            return BenchmarkResult.builder()
                .name(name)
                .algorithmName(algorithmName)
                .description(String.format("%s with %d folds", strategyName, numFolds))
                .accuracy(getMeanAccuracy())
                .precision(getMeanPrecision())
                .recall(getMeanRecall())
                .f1Score(getMeanF1())
                .addConfiguration("strategy", strategyName)
                .addConfiguration("folds", String.valueOf(numFolds))
                .addConfiguration("stdAccuracy", String.format("%.4f", getStdAccuracy()))
                .build();
        }

        /**
         * Generates a summary string.
         *
         * @return formatted summary
         */
        public String getSummary() {
            double[] ci = getAccuracyConfidenceInterval();
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("=== %s Results (%d folds) ===%n", strategyName, numFolds));
            sb.append(String.format("Accuracy:  %.4f +/- %.4f (95%% CI: [%.4f, %.4f])%n",
                getMeanAccuracy(), getStdAccuracy(), ci[0], ci[1]));
            sb.append(String.format("Precision: %.4f +/- %.4f%n", getMeanPrecision(), getStdPrecision()));
            sb.append(String.format("Recall:    %.4f +/- %.4f%n", getMeanRecall(), getStdRecall()));
            sb.append(String.format("F1 Score:  %.4f +/- %.4f%n", getMeanF1(), getStdF1()));
            sb.append(String.format("Total Time: %d ms%n", totalTimeMs));
            return sb.toString();
        }

        @Override
        public String toString() {
            return String.format("CrossValidationResult{strategy='%s', folds=%d, accuracy=%.4f +/- %.4f}",
                strategyName, numFolds, getMeanAccuracy(), getStdAccuracy());
        }
    }

    /**
     * Result of a single fold in cross-validation.
     */
    public static class FoldResult {
        private final int foldIndex;
        private final int trainSize;
        private final int testSize;
        private final int correctPredictions;
        private final double accuracy;
        private final double precision;
        private final double recall;
        private final double f1Score;
        private final long trainingTimeMs;
        private final long testingTimeMs;
        private final List<Prediction> predictions;

        public FoldResult(int foldIndex, int trainSize, int testSize, int correct,
                         double accuracy, double precision, double recall, double f1,
                         long trainTime, long testTime, List<Prediction> predictions) {
            this.foldIndex = foldIndex;
            this.trainSize = trainSize;
            this.testSize = testSize;
            this.correctPredictions = correct;
            this.accuracy = accuracy;
            this.precision = precision;
            this.recall = recall;
            this.f1Score = f1;
            this.trainingTimeMs = trainTime;
            this.testingTimeMs = testTime;
            this.predictions = predictions != null ? new ArrayList<>(predictions) : new ArrayList<>();
        }

        public int getFoldIndex() { return foldIndex; }
        public int getTrainSize() { return trainSize; }
        public int getTestSize() { return testSize; }
        public int getCorrectPredictions() { return correctPredictions; }
        public double getAccuracy() { return accuracy; }
        public double getPrecision() { return precision; }
        public double getRecall() { return recall; }
        public double getF1Score() { return f1Score; }
        public long getTrainingTimeMs() { return trainingTimeMs; }
        public long getTestingTimeMs() { return testingTimeMs; }
        public List<Prediction> getPredictions() { return Collections.unmodifiableList(predictions); }

        @Override
        public String toString() {
            return String.format("FoldResult{fold=%d, accuracy=%.4f, train=%d, test=%d}",
                foldIndex, accuracy, trainSize, testSize);
        }
    }

    /**
     * Represents a single prediction with actual and predicted labels.
     */
    public static class Prediction {
        private final String actualLabel;
        private final String predictedLabel;
        private final double confidence;
        private final double distance;

        public Prediction(String actual, String predicted, double confidence, double distance) {
            this.actualLabel = actual;
            this.predictedLabel = predicted;
            this.confidence = confidence;
            this.distance = distance;
        }

        public String getActualLabel() { return actualLabel; }
        public String getPredictedLabel() { return predictedLabel; }
        public double getConfidence() { return confidence; }
        public double getDistance() { return distance; }
        public boolean isCorrect() { return actualLabel.equals(predictedLabel); }

        @Override
        public String toString() {
            return String.format("Prediction{actual='%s', predicted='%s', conf=%.4f, correct=%s}",
                actualLabel, predictedLabel, confidence, isCorrect());
        }
    }

    /**
     * Creates a CrossValidator with the specified extractor and classifier suppliers.
     *
     * <p>Suppliers are used to create fresh instances for each fold to avoid
     * contamination between folds.</p>
     *
     * @param extractorSupplier supplier for creating feature extractors
     * @param classifierSupplier supplier for creating classifiers
     */
    public CrossValidator(Supplier<FeatureExtractor> extractorSupplier,
                         Supplier<FaceClassifier> classifierSupplier) {
        this(extractorSupplier, classifierSupplier, new CrossValidationConfig());
    }

    /**
     * Creates a CrossValidator with configuration.
     *
     * @param extractorSupplier supplier for creating feature extractors
     * @param classifierSupplier supplier for creating classifiers
     * @param config configuration options
     */
    public CrossValidator(Supplier<FeatureExtractor> extractorSupplier,
                         Supplier<FaceClassifier> classifierSupplier,
                         CrossValidationConfig config) {
        this.extractorSupplier = Objects.requireNonNull(extractorSupplier);
        this.classifierSupplier = Objects.requireNonNull(classifierSupplier);
        this.config = config;
    }

    /**
     * Performs K-fold cross validation.
     *
     * @param dataset the dataset to evaluate
     * @param k number of folds
     * @return cross validation result
     */
    public CrossValidationResult kFoldCrossValidation(DatasetLoader.LoadedDataset dataset, int k) {
        return kFoldCrossValidation(dataset, k, System.currentTimeMillis());
    }

    /**
     * Performs K-fold cross validation with a specific random seed.
     *
     * @param dataset the dataset to evaluate
     * @param k number of folds
     * @param seed random seed for reproducibility
     * @return cross validation result
     */
    public CrossValidationResult kFoldCrossValidation(DatasetLoader.LoadedDataset dataset, int k, long seed) {
        if (config.isVerbose()) {
            System.out.printf("Starting %d-fold cross validation on %s%n", k, dataset.getName());
        }

        long startTime = System.currentTimeMillis();
        List<DatasetLoader.DatasetSplit> folds = dataset.kFoldSplit(k, seed);
        List<FoldResult> results = new ArrayList<>();

        for (int i = 0; i < folds.size(); i++) {
            if (config.isVerbose()) {
                System.out.printf("  Processing fold %d/%d...%n", i + 1, k);
            }
            FoldResult result = evaluateFold(folds.get(i), i);
            results.add(result);
        }

        long totalTime = System.currentTimeMillis() - startTime;
        return new CrossValidationResult("K-Fold Cross Validation (k=" + k + ")", results, totalTime);
    }

    /**
     * Performs stratified K-fold cross validation.
     * Maintains class distribution in each fold.
     *
     * @param dataset the dataset to evaluate
     * @param k number of folds
     * @return cross validation result
     */
    public CrossValidationResult stratifiedKFold(DatasetLoader.LoadedDataset dataset, int k) {
        return stratifiedKFold(dataset, k, System.currentTimeMillis());
    }

    /**
     * Performs stratified K-fold cross validation with seed.
     *
     * @param dataset the dataset to evaluate
     * @param k number of folds
     * @param seed random seed
     * @return cross validation result
     */
    public CrossValidationResult stratifiedKFold(DatasetLoader.LoadedDataset dataset, int k, long seed) {
        if (config.isVerbose()) {
            System.out.printf("Starting stratified %d-fold cross validation on %s%n", k, dataset.getName());
        }

        long startTime = System.currentTimeMillis();
        List<DatasetLoader.DatasetSplit> folds = dataset.stratifiedKFold(k, seed);
        List<FoldResult> results = new ArrayList<>();

        for (int i = 0; i < folds.size(); i++) {
            if (config.isVerbose()) {
                System.out.printf("  Processing fold %d/%d...%n", i + 1, k);
            }
            FoldResult result = evaluateFold(folds.get(i), i);
            results.add(result);
        }

        long totalTime = System.currentTimeMillis() - startTime;
        return new CrossValidationResult("Stratified K-Fold (k=" + k + ")", results, totalTime);
    }

    /**
     * Performs leave-one-out cross validation.
     * Each sample is used as the test set once while remaining samples form training set.
     *
     * @param dataset the dataset to evaluate
     * @return cross validation result
     */
    public CrossValidationResult leaveOneOut(DatasetLoader.LoadedDataset dataset) {
        if (config.isVerbose()) {
            System.out.printf("Starting leave-one-out cross validation on %s (%d samples)%n",
                dataset.getName(), dataset.getFaces().size());
        }

        long startTime = System.currentTimeMillis();
        List<DatasetLoader.LabeledFace> allFaces = dataset.getFaces();
        List<FoldResult> results = new ArrayList<>();

        for (int i = 0; i < allFaces.size(); i++) {
            if (config.isVerbose() && i % 10 == 0) {
                System.out.printf("  Processing sample %d/%d...%n", i + 1, allFaces.size());
            }

            // Create train/test split with one sample left out
            List<DatasetLoader.LabeledFace> trainSet = new ArrayList<>();
            List<DatasetLoader.LabeledFace> testSet = new ArrayList<>();

            for (int j = 0; j < allFaces.size(); j++) {
                if (j == i) {
                    testSet.add(allFaces.get(j));
                } else {
                    trainSet.add(allFaces.get(j));
                }
            }

            DatasetLoader.DatasetSplit split = new DatasetLoader.DatasetSplit(
                trainSet, testSet, i, allFaces.size());
            FoldResult result = evaluateFold(split, i);
            results.add(result);
        }

        long totalTime = System.currentTimeMillis() - startTime;
        return new CrossValidationResult("Leave-One-Out", results, totalTime);
    }

    /**
     * Performs leave-P-out cross validation per class.
     * Leaves P samples out from each class.
     *
     * @param dataset the dataset to evaluate
     * @param p number of samples to leave out per class
     * @return cross validation result
     */
    public CrossValidationResult leavePOut(DatasetLoader.LoadedDataset dataset, int p) {
        if (config.isVerbose()) {
            System.out.printf("Starting leave-%d-out cross validation on %s%n", p, dataset.getName());
        }

        long startTime = System.currentTimeMillis();
        int imagesPerClass = dataset.getImagesPerClass();
        int numIterations = imagesPerClass / p;
        List<FoldResult> results = new ArrayList<>();

        for (int iter = 0; iter < numIterations; iter++) {
            if (config.isVerbose()) {
                System.out.printf("  Processing iteration %d/%d...%n", iter + 1, numIterations);
            }

            List<DatasetLoader.LabeledFace> trainSet = new ArrayList<>();
            List<DatasetLoader.LabeledFace> testSet = new ArrayList<>();

            for (String label : dataset.getLabels()) {
                List<DatasetLoader.LabeledFace> classFaces = dataset.getFacesByLabel(label);
                int startIdx = iter * p;
                int endIdx = Math.min(startIdx + p, classFaces.size());

                for (int i = 0; i < classFaces.size(); i++) {
                    if (i >= startIdx && i < endIdx) {
                        testSet.add(classFaces.get(i));
                    } else {
                        trainSet.add(classFaces.get(i));
                    }
                }
            }

            DatasetLoader.DatasetSplit split = new DatasetLoader.DatasetSplit(
                trainSet, testSet, iter, numIterations);
            FoldResult result = evaluateFold(split, iter);
            results.add(result);
        }

        long totalTime = System.currentTimeMillis() - startTime;
        return new CrossValidationResult("Leave-" + p + "-Out", results, totalTime);
    }

    /**
     * Performs repeated K-fold cross validation.
     * Runs K-fold CV multiple times with different random seeds.
     *
     * @param dataset the dataset to evaluate
     * @param k number of folds
     * @param repetitions number of times to repeat
     * @param baseSeed base random seed
     * @return aggregated cross validation result
     */
    public CrossValidationResult repeatedKFold(DatasetLoader.LoadedDataset dataset,
                                               int k, int repetitions, long baseSeed) {
        if (config.isVerbose()) {
            System.out.printf("Starting repeated %d-fold cross validation (%d reps) on %s%n",
                k, repetitions, dataset.getName());
        }

        long startTime = System.currentTimeMillis();
        List<FoldResult> allResults = new ArrayList<>();

        for (int rep = 0; rep < repetitions; rep++) {
            if (config.isVerbose()) {
                System.out.printf("  Repetition %d/%d...%n", rep + 1, repetitions);
            }

            List<DatasetLoader.DatasetSplit> folds = dataset.kFoldSplit(k, baseSeed + rep);
            for (int i = 0; i < folds.size(); i++) {
                FoldResult result = evaluateFold(folds.get(i), rep * k + i);
                allResults.add(result);
            }
        }

        long totalTime = System.currentTimeMillis() - startTime;
        return new CrossValidationResult(
            String.format("Repeated K-Fold (k=%d, rep=%d)", k, repetitions),
            allResults, totalTime);
    }

    /**
     * Evaluates a single fold.
     */
    private FoldResult evaluateFold(DatasetLoader.DatasetSplit split, int foldIndex) {
        // Create fresh instances
        FeatureExtractor extractor = extractorSupplier.get();
        FaceClassifier classifier = classifierSupplier.get();

        // Training phase
        long trainStart = System.currentTimeMillis();

        List<FaceImage> trainImages = split.getTrainImages();
        List<String> trainLabels = split.getTrainLabels();

        // Train the extractor
        extractor.train(trainImages, trainLabels);

        // Extract features and enroll identities
        Map<String, Identity> identities = new HashMap<>();
        for (int i = 0; i < trainImages.size(); i++) {
            FaceImage image = trainImages.get(i);
            String label = trainLabels.get(i);

            FeatureVector features = extractor.extract(image);

            Identity identity = identities.get(label);
            if (identity == null) {
                identity = new Identity(label);
                identities.put(label, identity);
            }
            identity.enrollSample(features, 1.0, "training");
        }

        // Enroll in classifier
        classifier.clear();
        for (Identity identity : identities.values()) {
            classifier.enroll(identity);
        }

        long trainTime = System.currentTimeMillis() - trainStart;

        // Testing phase
        long testStart = System.currentTimeMillis();
        List<Prediction> predictions = new ArrayList<>();
        int correct = 0;

        // Per-class counters for precision/recall
        Map<String, int[]> classStats = new HashMap<>(); // [TP, FP, FN]
        for (String label : identities.keySet()) {
            classStats.put(label, new int[3]);
        }

        for (DatasetLoader.LabeledFace testFace : split.getTestSet()) {
            FeatureVector features = extractor.extract(testFace.getImage());
            RecognitionResult result = classifier.classify(features, config.getRecognitionThreshold());

            String actualLabel = testFace.getLabel();
            String predictedLabel = result.getIdentity()
                .map(Identity::getName)
                .orElse("UNKNOWN");

            double confidence = result.getConfidence();
            double distance = result.getDistance();

            predictions.add(new Prediction(actualLabel, predictedLabel, confidence, distance));

            if (actualLabel.equals(predictedLabel)) {
                correct++;
                // True positive for this class
                if (classStats.containsKey(actualLabel)) {
                    classStats.get(actualLabel)[0]++;
                }
            } else {
                // False negative for actual class
                if (classStats.containsKey(actualLabel)) {
                    classStats.get(actualLabel)[2]++;
                }
                // False positive for predicted class (if not UNKNOWN)
                if (classStats.containsKey(predictedLabel)) {
                    classStats.get(predictedLabel)[1]++;
                }
            }
        }

        long testTime = System.currentTimeMillis() - testStart;

        // Calculate metrics
        double accuracy = split.getTestSize() > 0 ? (double) correct / split.getTestSize() : 0.0;

        // Macro-averaged precision and recall
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

        return new FoldResult(foldIndex, split.getTrainSize(), split.getTestSize(),
            correct, accuracy, precision, recall, f1, trainTime, testTime,
            config.isCollectPerFoldResults() ? predictions : null);
    }

    /**
     * Computes a confusion matrix from cross-validation results.
     *
     * @param cvResult the cross-validation result
     * @return confusion matrix
     */
    public BenchmarkResult.ConfusionMatrix computeConfusionMatrix(CrossValidationResult cvResult) {
        // Collect all unique labels
        Set<String> labelSet = new TreeSet<>();
        for (FoldResult fold : cvResult.getFoldResults()) {
            for (Prediction pred : fold.getPredictions()) {
                labelSet.add(pred.getActualLabel());
                if (!"UNKNOWN".equals(pred.getPredictedLabel())) {
                    labelSet.add(pred.getPredictedLabel());
                }
            }
        }

        List<String> labels = new ArrayList<>(labelSet);
        Map<String, Integer> labelIndex = new HashMap<>();
        for (int i = 0; i < labels.size(); i++) {
            labelIndex.put(labels.get(i), i);
        }

        // Build confusion matrix
        int[][] matrix = new int[labels.size()][labels.size()];

        for (FoldResult fold : cvResult.getFoldResults()) {
            for (Prediction pred : fold.getPredictions()) {
                Integer actualIdx = labelIndex.get(pred.getActualLabel());
                Integer predictedIdx = labelIndex.get(pred.getPredictedLabel());

                if (actualIdx != null && predictedIdx != null) {
                    matrix[actualIdx][predictedIdx]++;
                }
            }
        }

        return new BenchmarkResult.ConfusionMatrix(labels, matrix);
    }

    /**
     * Compares multiple algorithms using cross-validation.
     *
     * @param dataset the dataset to use
     * @param extractorSuppliers map of algorithm name to extractor supplier
     * @param classifierSupplier classifier supplier (shared)
     * @param k number of folds
     * @param seed random seed
     * @return map of algorithm name to cross-validation result
     */
    public static Map<String, CrossValidationResult> compareAlgorithms(
            DatasetLoader.LoadedDataset dataset,
            Map<String, Supplier<FeatureExtractor>> extractorSuppliers,
            Supplier<FaceClassifier> classifierSupplier,
            int k, long seed) {

        Map<String, CrossValidationResult> results = new LinkedHashMap<>();

        for (Map.Entry<String, Supplier<FeatureExtractor>> entry : extractorSuppliers.entrySet()) {
            String algName = entry.getKey();
            System.out.printf("%nEvaluating %s...%n", algName);

            CrossValidator validator = new CrossValidator(entry.getValue(), classifierSupplier);
            CrossValidationResult result = validator.stratifiedKFold(dataset, k, seed);
            results.put(algName, result);

            System.out.printf("  Accuracy: %.4f +/- %.4f%n",
                result.getMeanAccuracy(), result.getStdAccuracy());
        }

        return results;
    }

    /**
     * Performs statistical significance test (paired t-test) between two algorithms.
     *
     * @param result1 first algorithm result
     * @param result2 second algorithm result
     * @return p-value of the test
     */
    public static double pairedTTest(CrossValidationResult result1, CrossValidationResult result2) {
        List<FoldResult> folds1 = result1.getFoldResults();
        List<FoldResult> folds2 = result2.getFoldResults();

        if (folds1.size() != folds2.size()) {
            throw new IllegalArgumentException("Results must have same number of folds");
        }

        int n = folds1.size();
        double[] differences = new double[n];
        double sumDiff = 0;

        for (int i = 0; i < n; i++) {
            differences[i] = folds1.get(i).getAccuracy() - folds2.get(i).getAccuracy();
            sumDiff += differences[i];
        }

        double meanDiff = sumDiff / n;
        double varDiff = 0;
        for (double d : differences) {
            varDiff += Math.pow(d - meanDiff, 2);
        }
        varDiff /= (n - 1);

        double t = meanDiff / Math.sqrt(varDiff / n);

        // Approximate p-value using Student's t distribution
        // This is a simplified approximation; use a proper statistical library for production
        double df = n - 1;
        double pValue = approximateTDistributionPValue(Math.abs(t), df);

        return 2 * pValue; // Two-tailed test
    }

    /**
     * Approximates p-value from t-distribution (simple approximation).
     */
    private static double approximateTDistributionPValue(double t, double df) {
        // Using approximation: p ≈ 2 * (1 - cdf(t))
        // For large df, t-distribution approaches normal distribution
        if (df > 30) {
            // Use normal approximation
            return 0.5 * (1 - erf(t / Math.sqrt(2)));
        }

        // Simple approximation for smaller df
        double x = df / (df + t * t);
        double beta = 0.5; // Incomplete beta function approximation
        return 0.5 * Math.pow(x, df / 2);
    }

    /**
     * Error function approximation.
     */
    private static double erf(double x) {
        // Abramowitz and Stegun approximation
        double t = 1.0 / (1.0 + 0.5 * Math.abs(x));
        double tau = t * Math.exp(-x * x - 1.26551223 +
            t * (1.00002368 +
            t * (0.37409196 +
            t * (0.09678418 +
            t * (-0.18628806 +
            t * (0.27886807 +
            t * (-1.13520398 +
            t * (1.48851587 +
            t * (-0.82215223 +
            t * 0.17087277)))))))));
        return x >= 0 ? 1 - tau : tau - 1;
    }
}
