package com.facerecognition.benchmark;

import java.io.Serializable;
import java.time.Duration;
import java.time.LocalDateTime;
import java.util.*;

/**
 * Container for benchmark results including accuracy metrics, timing statistics,
 * and detailed analysis data.
 *
 * <p>This class captures comprehensive benchmark results including:</p>
 * <ul>
 *   <li><b>Classification Metrics</b>: Accuracy, precision, recall, F1-score</li>
 *   <li><b>Confusion Matrix</b>: True/false positives and negatives per class</li>
 *   <li><b>ROC Curve Data</b>: Points for plotting Receiver Operating Characteristic curves</li>
 *   <li><b>Error Rate Metrics</b>: FAR, FRR, and EER (Equal Error Rate)</li>
 *   <li><b>Performance Metrics</b>: Processing times and memory usage</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * BenchmarkResult result = BenchmarkResult.builder()
 *     .name("Eigenfaces-ORL")
 *     .accuracy(0.95)
 *     .precision(0.94)
 *     .recall(0.96)
 *     .f1Score(0.95)
 *     .build();
 *
 * System.out.println("Accuracy: " + result.getAccuracy());
 * System.out.println("EER: " + result.getEqualErrorRate());
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see AccuracyBenchmark
 * @see PerformanceBenchmark
 */
public class BenchmarkResult implements Serializable {

    private static final long serialVersionUID = 1L;

    // Identification
    private final String id;
    private final String name;
    private final String description;
    private final LocalDateTime timestamp;
    private final String algorithmName;
    private final Map<String, String> configuration;

    // Classification Metrics
    private final double accuracy;
    private final double precision;
    private final double recall;
    private final double f1Score;
    private final double specificity;

    // Error Rate Metrics
    private final double falseAcceptRate;       // FAR
    private final double falseRejectRate;       // FRR
    private final double equalErrorRate;        // EER
    private final double areaUnderCurve;        // AUC-ROC

    // Confusion Matrix
    private final ConfusionMatrix confusionMatrix;

    // ROC Curve Data
    private final List<RocPoint> rocCurve;

    // DET Curve Data (Detection Error Tradeoff)
    private final List<DetPoint> detCurve;

    // Performance Metrics
    private final PerformanceMetrics performanceMetrics;

    // Per-class metrics
    private final Map<String, ClassMetrics> perClassMetrics;

    // Dataset information
    private final DatasetInfo datasetInfo;

    /**
     * Represents a single point on the ROC curve.
     */
    public static class RocPoint implements Serializable {
        private static final long serialVersionUID = 1L;

        private final double threshold;
        private final double truePositiveRate;
        private final double falsePositiveRate;

        public RocPoint(double threshold, double tpr, double fpr) {
            this.threshold = threshold;
            this.truePositiveRate = tpr;
            this.falsePositiveRate = fpr;
        }

        public double getThreshold() { return threshold; }
        public double getTruePositiveRate() { return truePositiveRate; }
        public double getFalsePositiveRate() { return falsePositiveRate; }

        @Override
        public String toString() {
            return String.format("RocPoint{threshold=%.4f, TPR=%.4f, FPR=%.4f}",
                threshold, truePositiveRate, falsePositiveRate);
        }
    }

    /**
     * Represents a single point on the DET (Detection Error Tradeoff) curve.
     */
    public static class DetPoint implements Serializable {
        private static final long serialVersionUID = 1L;

        private final double threshold;
        private final double falseRejectRate;
        private final double falseAcceptRate;

        public DetPoint(double threshold, double frr, double far) {
            this.threshold = threshold;
            this.falseRejectRate = frr;
            this.falseAcceptRate = far;
        }

        public double getThreshold() { return threshold; }
        public double getFalseRejectRate() { return falseRejectRate; }
        public double getFalseAcceptRate() { return falseAcceptRate; }

        @Override
        public String toString() {
            return String.format("DetPoint{threshold=%.4f, FRR=%.4f, FAR=%.4f}",
                threshold, falseRejectRate, falseAcceptRate);
        }
    }

    /**
     * Confusion matrix for multi-class classification.
     */
    public static class ConfusionMatrix implements Serializable {
        private static final long serialVersionUID = 1L;

        private final List<String> labels;
        private final int[][] matrix;
        private final int totalSamples;

        public ConfusionMatrix(List<String> labels, int[][] matrix) {
            this.labels = new ArrayList<>(labels);
            this.matrix = deepCopy(matrix);

            int total = 0;
            for (int[] row : matrix) {
                for (int val : row) {
                    total += val;
                }
            }
            this.totalSamples = total;
        }

        private int[][] deepCopy(int[][] original) {
            int[][] copy = new int[original.length][];
            for (int i = 0; i < original.length; i++) {
                copy[i] = Arrays.copyOf(original[i], original[i].length);
            }
            return copy;
        }

        public List<String> getLabels() { return Collections.unmodifiableList(labels); }
        public int[][] getMatrix() { return deepCopy(matrix); }
        public int getTotalSamples() { return totalSamples; }
        public int getNumClasses() { return labels.size(); }

        /**
         * Gets the value at a specific position.
         *
         * @param actualIndex index of actual class
         * @param predictedIndex index of predicted class
         * @return count at the position
         */
        public int getValue(int actualIndex, int predictedIndex) {
            return matrix[actualIndex][predictedIndex];
        }

        /**
         * Gets the value for specific class labels.
         *
         * @param actualLabel actual class label
         * @param predictedLabel predicted class label
         * @return count for the combination
         */
        public int getValue(String actualLabel, String predictedLabel) {
            int actualIdx = labels.indexOf(actualLabel);
            int predictedIdx = labels.indexOf(predictedLabel);
            if (actualIdx < 0 || predictedIdx < 0) {
                throw new IllegalArgumentException("Unknown label");
            }
            return matrix[actualIdx][predictedIdx];
        }

        /**
         * Gets the true positive count for a class.
         *
         * @param classIndex the class index
         * @return true positive count
         */
        public int getTruePositives(int classIndex) {
            return matrix[classIndex][classIndex];
        }

        /**
         * Gets the false positive count for a class.
         *
         * @param classIndex the class index
         * @return false positive count
         */
        public int getFalsePositives(int classIndex) {
            int fp = 0;
            for (int i = 0; i < matrix.length; i++) {
                if (i != classIndex) {
                    fp += matrix[i][classIndex];
                }
            }
            return fp;
        }

        /**
         * Gets the false negative count for a class.
         *
         * @param classIndex the class index
         * @return false negative count
         */
        public int getFalseNegatives(int classIndex) {
            int fn = 0;
            for (int j = 0; j < matrix[classIndex].length; j++) {
                if (j != classIndex) {
                    fn += matrix[classIndex][j];
                }
            }
            return fn;
        }

        /**
         * Gets the true negative count for a class.
         *
         * @param classIndex the class index
         * @return true negative count
         */
        public int getTrueNegatives(int classIndex) {
            return totalSamples - getTruePositives(classIndex)
                   - getFalsePositives(classIndex) - getFalseNegatives(classIndex);
        }

        /**
         * Computes accuracy from the confusion matrix.
         *
         * @return the accuracy (0.0 to 1.0)
         */
        public double computeAccuracy() {
            int correct = 0;
            for (int i = 0; i < matrix.length; i++) {
                correct += matrix[i][i];
            }
            return totalSamples > 0 ? (double) correct / totalSamples : 0.0;
        }

        /**
         * Generates a formatted string representation.
         *
         * @return formatted confusion matrix
         */
        public String toFormattedString() {
            StringBuilder sb = new StringBuilder();
            int maxLabelLen = labels.stream().mapToInt(String::length).max().orElse(10);
            maxLabelLen = Math.max(maxLabelLen, 8);

            // Header
            sb.append(String.format("%" + maxLabelLen + "s", "Actual\\Pred"));
            for (String label : labels) {
                sb.append(String.format(" %8s", truncate(label, 8)));
            }
            sb.append("\n");

            // Rows
            for (int i = 0; i < matrix.length; i++) {
                sb.append(String.format("%" + maxLabelLen + "s", truncate(labels.get(i), maxLabelLen)));
                for (int j = 0; j < matrix[i].length; j++) {
                    sb.append(String.format(" %8d", matrix[i][j]));
                }
                sb.append("\n");
            }

            return sb.toString();
        }

        private String truncate(String s, int maxLen) {
            return s.length() <= maxLen ? s : s.substring(0, maxLen - 2) + "..";
        }

        @Override
        public String toString() {
            return String.format("ConfusionMatrix{classes=%d, samples=%d, accuracy=%.2f%%}",
                labels.size(), totalSamples, computeAccuracy() * 100);
        }
    }

    /**
     * Performance metrics including timing and memory usage.
     */
    public static class PerformanceMetrics implements Serializable {
        private static final long serialVersionUID = 1L;

        private final TimingStats detectionTime;
        private final TimingStats extractionTime;
        private final TimingStats matchingTime;
        private final TimingStats totalTime;
        private final long peakMemoryUsage;
        private final long averageMemoryUsage;
        private final int totalSamplesProcessed;
        private final double throughput;  // samples per second

        public PerformanceMetrics(TimingStats detectionTime, TimingStats extractionTime,
                                  TimingStats matchingTime, TimingStats totalTime,
                                  long peakMemory, long avgMemory,
                                  int samples, double throughput) {
            this.detectionTime = detectionTime;
            this.extractionTime = extractionTime;
            this.matchingTime = matchingTime;
            this.totalTime = totalTime;
            this.peakMemoryUsage = peakMemory;
            this.averageMemoryUsage = avgMemory;
            this.totalSamplesProcessed = samples;
            this.throughput = throughput;
        }

        public TimingStats getDetectionTime() { return detectionTime; }
        public TimingStats getExtractionTime() { return extractionTime; }
        public TimingStats getMatchingTime() { return matchingTime; }
        public TimingStats getTotalTime() { return totalTime; }
        public long getPeakMemoryUsage() { return peakMemoryUsage; }
        public long getAverageMemoryUsage() { return averageMemoryUsage; }
        public int getTotalSamplesProcessed() { return totalSamplesProcessed; }
        public double getThroughput() { return throughput; }

        /**
         * Gets peak memory usage in MB.
         *
         * @return memory in megabytes
         */
        public double getPeakMemoryMB() {
            return peakMemoryUsage / (1024.0 * 1024.0);
        }

        @Override
        public String toString() {
            return String.format("PerformanceMetrics{total=%.2fms, throughput=%.2f/s, peakMem=%.2fMB}",
                totalTime != null ? totalTime.getMean() : 0,
                throughput,
                getPeakMemoryMB());
        }
    }

    /**
     * Statistics for timing measurements.
     */
    public static class TimingStats implements Serializable {
        private static final long serialVersionUID = 1L;

        private final double min;
        private final double max;
        private final double mean;
        private final double median;
        private final double stdDev;
        private final double p95;  // 95th percentile
        private final double p99;  // 99th percentile
        private final int count;

        public TimingStats(double min, double max, double mean, double median,
                          double stdDev, double p95, double p99, int count) {
            this.min = min;
            this.max = max;
            this.mean = mean;
            this.median = median;
            this.stdDev = stdDev;
            this.p95 = p95;
            this.p99 = p99;
            this.count = count;
        }

        /**
         * Creates TimingStats from a list of measurements.
         *
         * @param values the timing values in milliseconds
         * @return computed statistics
         */
        public static TimingStats fromValues(List<Double> values) {
            if (values == null || values.isEmpty()) {
                return new TimingStats(0, 0, 0, 0, 0, 0, 0, 0);
            }

            List<Double> sorted = new ArrayList<>(values);
            Collections.sort(sorted);

            int n = sorted.size();
            double min = sorted.get(0);
            double max = sorted.get(n - 1);
            double sum = sorted.stream().mapToDouble(Double::doubleValue).sum();
            double mean = sum / n;

            double median = n % 2 == 0 ?
                (sorted.get(n / 2 - 1) + sorted.get(n / 2)) / 2 :
                sorted.get(n / 2);

            double variance = sorted.stream()
                .mapToDouble(v -> Math.pow(v - mean, 2))
                .sum() / n;
            double stdDev = Math.sqrt(variance);

            double p95 = sorted.get((int) Math.floor(0.95 * (n - 1)));
            double p99 = sorted.get((int) Math.floor(0.99 * (n - 1)));

            return new TimingStats(min, max, mean, median, stdDev, p95, p99, n);
        }

        public double getMin() { return min; }
        public double getMax() { return max; }
        public double getMean() { return mean; }
        public double getMedian() { return median; }
        public double getStdDev() { return stdDev; }
        public double getP95() { return p95; }
        public double getP99() { return p99; }
        public int getCount() { return count; }

        @Override
        public String toString() {
            return String.format("TimingStats{mean=%.2fms, std=%.2fms, p95=%.2fms}",
                mean, stdDev, p95);
        }
    }

    /**
     * Per-class metrics for detailed analysis.
     */
    public static class ClassMetrics implements Serializable {
        private static final long serialVersionUID = 1L;

        private final String className;
        private final int samples;
        private final double precision;
        private final double recall;
        private final double f1Score;
        private final double specificity;
        private final int truePositives;
        private final int falsePositives;
        private final int falseNegatives;
        private final int trueNegatives;

        public ClassMetrics(String className, int samples, double precision,
                           double recall, double f1Score, double specificity,
                           int tp, int fp, int fn, int tn) {
            this.className = className;
            this.samples = samples;
            this.precision = precision;
            this.recall = recall;
            this.f1Score = f1Score;
            this.specificity = specificity;
            this.truePositives = tp;
            this.falsePositives = fp;
            this.falseNegatives = fn;
            this.trueNegatives = tn;
        }

        public String getClassName() { return className; }
        public int getSamples() { return samples; }
        public double getPrecision() { return precision; }
        public double getRecall() { return recall; }
        public double getF1Score() { return f1Score; }
        public double getSpecificity() { return specificity; }
        public int getTruePositives() { return truePositives; }
        public int getFalsePositives() { return falsePositives; }
        public int getFalseNegatives() { return falseNegatives; }
        public int getTrueNegatives() { return trueNegatives; }

        @Override
        public String toString() {
            return String.format("ClassMetrics{class='%s', precision=%.4f, recall=%.4f, f1=%.4f}",
                className, precision, recall, f1Score);
        }
    }

    /**
     * Information about the dataset used in benchmarking.
     */
    public static class DatasetInfo implements Serializable {
        private static final long serialVersionUID = 1L;

        private final String name;
        private final String format;
        private final int totalImages;
        private final int numClasses;
        private final int imagesPerClass;
        private final int imageWidth;
        private final int imageHeight;
        private final int trainSize;
        private final int testSize;

        public DatasetInfo(String name, String format, int totalImages, int numClasses,
                          int imagesPerClass, int imageWidth, int imageHeight,
                          int trainSize, int testSize) {
            this.name = name;
            this.format = format;
            this.totalImages = totalImages;
            this.numClasses = numClasses;
            this.imagesPerClass = imagesPerClass;
            this.imageWidth = imageWidth;
            this.imageHeight = imageHeight;
            this.trainSize = trainSize;
            this.testSize = testSize;
        }

        public String getName() { return name; }
        public String getFormat() { return format; }
        public int getTotalImages() { return totalImages; }
        public int getNumClasses() { return numClasses; }
        public int getImagesPerClass() { return imagesPerClass; }
        public int getImageWidth() { return imageWidth; }
        public int getImageHeight() { return imageHeight; }
        public int getTrainSize() { return trainSize; }
        public int getTestSize() { return testSize; }

        @Override
        public String toString() {
            return String.format("DatasetInfo{name='%s', classes=%d, images=%d, train=%d, test=%d}",
                name, numClasses, totalImages, trainSize, testSize);
        }
    }

    /**
     * Builder for creating BenchmarkResult instances.
     */
    public static class Builder {
        private String name = "Unnamed Benchmark";
        private String description = "";
        private String algorithmName = "Unknown";
        private Map<String, String> configuration = new HashMap<>();

        private double accuracy = 0.0;
        private double precision = 0.0;
        private double recall = 0.0;
        private double f1Score = 0.0;
        private double specificity = 0.0;

        private double falseAcceptRate = 0.0;
        private double falseRejectRate = 0.0;
        private double equalErrorRate = 0.0;
        private double areaUnderCurve = 0.0;

        private ConfusionMatrix confusionMatrix;
        private List<RocPoint> rocCurve = new ArrayList<>();
        private List<DetPoint> detCurve = new ArrayList<>();
        private PerformanceMetrics performanceMetrics;
        private Map<String, ClassMetrics> perClassMetrics = new HashMap<>();
        private DatasetInfo datasetInfo;

        public Builder name(String name) {
            this.name = name;
            return this;
        }

        public Builder description(String description) {
            this.description = description;
            return this;
        }

        public Builder algorithmName(String algorithmName) {
            this.algorithmName = algorithmName;
            return this;
        }

        public Builder configuration(Map<String, String> configuration) {
            this.configuration = new HashMap<>(configuration);
            return this;
        }

        public Builder addConfiguration(String key, String value) {
            this.configuration.put(key, value);
            return this;
        }

        public Builder accuracy(double accuracy) {
            this.accuracy = accuracy;
            return this;
        }

        public Builder precision(double precision) {
            this.precision = precision;
            return this;
        }

        public Builder recall(double recall) {
            this.recall = recall;
            return this;
        }

        public Builder f1Score(double f1Score) {
            this.f1Score = f1Score;
            return this;
        }

        public Builder specificity(double specificity) {
            this.specificity = specificity;
            return this;
        }

        public Builder falseAcceptRate(double far) {
            this.falseAcceptRate = far;
            return this;
        }

        public Builder falseRejectRate(double frr) {
            this.falseRejectRate = frr;
            return this;
        }

        public Builder equalErrorRate(double eer) {
            this.equalErrorRate = eer;
            return this;
        }

        public Builder areaUnderCurve(double auc) {
            this.areaUnderCurve = auc;
            return this;
        }

        public Builder confusionMatrix(ConfusionMatrix matrix) {
            this.confusionMatrix = matrix;
            return this;
        }

        public Builder rocCurve(List<RocPoint> rocCurve) {
            this.rocCurve = new ArrayList<>(rocCurve);
            return this;
        }

        public Builder addRocPoint(double threshold, double tpr, double fpr) {
            this.rocCurve.add(new RocPoint(threshold, tpr, fpr));
            return this;
        }

        public Builder detCurve(List<DetPoint> detCurve) {
            this.detCurve = new ArrayList<>(detCurve);
            return this;
        }

        public Builder addDetPoint(double threshold, double frr, double far) {
            this.detCurve.add(new DetPoint(threshold, frr, far));
            return this;
        }

        public Builder performanceMetrics(PerformanceMetrics metrics) {
            this.performanceMetrics = metrics;
            return this;
        }

        public Builder perClassMetrics(Map<String, ClassMetrics> metrics) {
            this.perClassMetrics = new HashMap<>(metrics);
            return this;
        }

        public Builder addClassMetrics(ClassMetrics metrics) {
            this.perClassMetrics.put(metrics.getClassName(), metrics);
            return this;
        }

        public Builder datasetInfo(DatasetInfo info) {
            this.datasetInfo = info;
            return this;
        }

        /**
         * Computes metrics automatically from the confusion matrix if available.
         *
         * @return this builder
         */
        public Builder computeMetricsFromConfusionMatrix() {
            if (confusionMatrix != null) {
                this.accuracy = confusionMatrix.computeAccuracy();

                // Compute macro-averaged precision, recall, F1
                int numClasses = confusionMatrix.getNumClasses();
                double totalPrecision = 0, totalRecall = 0, totalF1 = 0, totalSpecificity = 0;

                for (int i = 0; i < numClasses; i++) {
                    int tp = confusionMatrix.getTruePositives(i);
                    int fp = confusionMatrix.getFalsePositives(i);
                    int fn = confusionMatrix.getFalseNegatives(i);
                    int tn = confusionMatrix.getTrueNegatives(i);

                    double p = tp + fp > 0 ? (double) tp / (tp + fp) : 0;
                    double r = tp + fn > 0 ? (double) tp / (tp + fn) : 0;
                    double f1 = p + r > 0 ? 2 * p * r / (p + r) : 0;
                    double spec = tn + fp > 0 ? (double) tn / (tn + fp) : 0;

                    totalPrecision += p;
                    totalRecall += r;
                    totalF1 += f1;
                    totalSpecificity += spec;

                    String className = confusionMatrix.getLabels().get(i);
                    int samples = tp + fn;
                    perClassMetrics.put(className,
                        new ClassMetrics(className, samples, p, r, f1, spec, tp, fp, fn, tn));
                }

                this.precision = totalPrecision / numClasses;
                this.recall = totalRecall / numClasses;
                this.f1Score = totalF1 / numClasses;
                this.specificity = totalSpecificity / numClasses;
            }
            return this;
        }

        public BenchmarkResult build() {
            return new BenchmarkResult(this);
        }
    }

    private BenchmarkResult(Builder builder) {
        this.id = UUID.randomUUID().toString();
        this.name = builder.name;
        this.description = builder.description;
        this.timestamp = LocalDateTime.now();
        this.algorithmName = builder.algorithmName;
        this.configuration = Collections.unmodifiableMap(new HashMap<>(builder.configuration));

        this.accuracy = builder.accuracy;
        this.precision = builder.precision;
        this.recall = builder.recall;
        this.f1Score = builder.f1Score;
        this.specificity = builder.specificity;

        this.falseAcceptRate = builder.falseAcceptRate;
        this.falseRejectRate = builder.falseRejectRate;
        this.equalErrorRate = builder.equalErrorRate;
        this.areaUnderCurve = builder.areaUnderCurve;

        this.confusionMatrix = builder.confusionMatrix;
        this.rocCurve = Collections.unmodifiableList(new ArrayList<>(builder.rocCurve));
        this.detCurve = Collections.unmodifiableList(new ArrayList<>(builder.detCurve));
        this.performanceMetrics = builder.performanceMetrics;
        this.perClassMetrics = Collections.unmodifiableMap(new HashMap<>(builder.perClassMetrics));
        this.datasetInfo = builder.datasetInfo;
    }

    /**
     * Creates a new builder.
     *
     * @return a new Builder instance
     */
    public static Builder builder() {
        return new Builder();
    }

    // Getters

    public String getId() { return id; }
    public String getName() { return name; }
    public String getDescription() { return description; }
    public LocalDateTime getTimestamp() { return timestamp; }
    public String getAlgorithmName() { return algorithmName; }
    public Map<String, String> getConfiguration() { return configuration; }

    public double getAccuracy() { return accuracy; }
    public double getPrecision() { return precision; }
    public double getRecall() { return recall; }
    public double getF1Score() { return f1Score; }
    public double getSpecificity() { return specificity; }

    public double getFalseAcceptRate() { return falseAcceptRate; }
    public double getFalseRejectRate() { return falseRejectRate; }
    public double getEqualErrorRate() { return equalErrorRate; }
    public double getAreaUnderCurve() { return areaUnderCurve; }

    public Optional<ConfusionMatrix> getConfusionMatrix() {
        return Optional.ofNullable(confusionMatrix);
    }

    public List<RocPoint> getRocCurve() { return rocCurve; }
    public List<DetPoint> getDetCurve() { return detCurve; }

    public Optional<PerformanceMetrics> getPerformanceMetrics() {
        return Optional.ofNullable(performanceMetrics);
    }

    public Map<String, ClassMetrics> getPerClassMetrics() { return perClassMetrics; }

    public Optional<DatasetInfo> getDatasetInfo() {
        return Optional.ofNullable(datasetInfo);
    }

    /**
     * Gets the accuracy as a percentage.
     *
     * @return accuracy percentage (0-100)
     */
    public double getAccuracyPercent() {
        return accuracy * 100;
    }

    /**
     * Gets the EER as a percentage.
     *
     * @return EER percentage (0-100)
     */
    public double getEerPercent() {
        return equalErrorRate * 100;
    }

    /**
     * Computes and returns the AUC if ROC data is available.
     *
     * @return computed AUC or stored value
     */
    public double computeAuc() {
        if (rocCurve.isEmpty()) {
            return areaUnderCurve;
        }

        // Trapezoidal rule for AUC computation
        List<RocPoint> sorted = new ArrayList<>(rocCurve);
        sorted.sort(Comparator.comparingDouble(RocPoint::getFalsePositiveRate));

        double auc = 0.0;
        for (int i = 1; i < sorted.size(); i++) {
            RocPoint p1 = sorted.get(i - 1);
            RocPoint p2 = sorted.get(i);
            double width = p2.getFalsePositiveRate() - p1.getFalsePositiveRate();
            double avgHeight = (p1.getTruePositiveRate() + p2.getTruePositiveRate()) / 2;
            auc += width * avgHeight;
        }

        return auc;
    }

    /**
     * Generates a summary string of the benchmark results.
     *
     * @return formatted summary
     */
    public String getSummary() {
        StringBuilder sb = new StringBuilder();
        sb.append("=== Benchmark Results: ").append(name).append(" ===\n");
        sb.append("Algorithm: ").append(algorithmName).append("\n");
        sb.append("Timestamp: ").append(timestamp).append("\n\n");

        sb.append("--- Classification Metrics ---\n");
        sb.append(String.format("Accuracy:    %.4f (%.2f%%)\n", accuracy, accuracy * 100));
        sb.append(String.format("Precision:   %.4f\n", precision));
        sb.append(String.format("Recall:      %.4f\n", recall));
        sb.append(String.format("F1-Score:    %.4f\n", f1Score));
        sb.append(String.format("Specificity: %.4f\n", specificity));
        sb.append("\n");

        sb.append("--- Error Rates ---\n");
        sb.append(String.format("FAR: %.4f (%.2f%%)\n", falseAcceptRate, falseAcceptRate * 100));
        sb.append(String.format("FRR: %.4f (%.2f%%)\n", falseRejectRate, falseRejectRate * 100));
        sb.append(String.format("EER: %.4f (%.2f%%)\n", equalErrorRate, equalErrorRate * 100));
        sb.append(String.format("AUC: %.4f\n", areaUnderCurve > 0 ? areaUnderCurve : computeAuc()));

        if (performanceMetrics != null) {
            sb.append("\n--- Performance ---\n");
            sb.append(String.format("Throughput: %.2f samples/sec\n", performanceMetrics.getThroughput()));
            if (performanceMetrics.getTotalTime() != null) {
                sb.append(String.format("Avg Time:   %.2f ms\n", performanceMetrics.getTotalTime().getMean()));
            }
            sb.append(String.format("Peak Memory: %.2f MB\n", performanceMetrics.getPeakMemoryMB()));
        }

        if (datasetInfo != null) {
            sb.append("\n--- Dataset ---\n");
            sb.append(String.format("Name: %s (%s)\n", datasetInfo.getName(), datasetInfo.getFormat()));
            sb.append(String.format("Classes: %d, Total Images: %d\n",
                datasetInfo.getNumClasses(), datasetInfo.getTotalImages()));
            sb.append(String.format("Train/Test Split: %d/%d\n",
                datasetInfo.getTrainSize(), datasetInfo.getTestSize()));
        }

        return sb.toString();
    }

    @Override
    public String toString() {
        return String.format("BenchmarkResult{name='%s', algorithm='%s', accuracy=%.4f, f1=%.4f, eer=%.4f}",
            name, algorithmName, accuracy, f1Score, equalErrorRate);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        BenchmarkResult that = (BenchmarkResult) o;
        return id.equals(that.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }
}
