package com.facerecognition.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;

import java.io.*;
import java.nio.file.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.*;

/**
 * Report generator for benchmark results.
 *
 * <p>Supports multiple output formats:</p>
 * <ul>
 *   <li><b>Markdown</b>: Human-readable reports with tables</li>
 *   <li><b>JSON</b>: Machine-readable structured data</li>
 *   <li><b>HTML</b>: Interactive reports with charts</li>
 *   <li><b>CSV</b>: Spreadsheet-compatible data export</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * ReportGenerator generator = new ReportGenerator();
 *
 * // Generate Markdown report
 * String markdown = generator.generateMarkdown(result);
 *
 * // Generate JSON and save to file
 * generator.saveJson(results, Paths.get("benchmark_results.json"));
 *
 * // Generate HTML report with charts
 * generator.saveHtml(results, Paths.get("report.html"));
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see BenchmarkResult
 */
public class ReportGenerator {

    private static final DateTimeFormatter DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");
    private final ObjectMapper objectMapper;

    /**
     * Creates a new ReportGenerator.
     */
    public ReportGenerator() {
        this.objectMapper = new ObjectMapper()
            .registerModule(new JavaTimeModule())
            .enable(SerializationFeature.INDENT_OUTPUT)
            .disable(SerializationFeature.WRITE_DATES_AS_TIMESTAMPS);
    }

    // ==================== Markdown Generation ====================

    /**
     * Generates a Markdown report for a single benchmark result.
     *
     * @param result the benchmark result
     * @return formatted Markdown string
     */
    public String generateMarkdown(BenchmarkResult result) {
        StringBuilder sb = new StringBuilder();

        // Header
        sb.append("# ").append(result.getName()).append("\n\n");
        sb.append("**Algorithm:** ").append(result.getAlgorithmName()).append("\n");
        sb.append("**Date:** ").append(result.getTimestamp().format(DATE_FORMAT)).append("\n");
        if (!result.getDescription().isEmpty()) {
            sb.append("**Description:** ").append(result.getDescription()).append("\n");
        }
        sb.append("\n");

        // Dataset Info
        result.getDatasetInfo().ifPresent(info -> {
            sb.append("## Dataset\n\n");
            sb.append("| Property | Value |\n");
            sb.append("|----------|-------|\n");
            sb.append(String.format("| Name | %s |\n", info.getName()));
            sb.append(String.format("| Format | %s |\n", info.getFormat()));
            sb.append(String.format("| Classes | %d |\n", info.getNumClasses()));
            sb.append(String.format("| Total Images | %d |\n", info.getTotalImages()));
            sb.append(String.format("| Images/Class | %d |\n", info.getImagesPerClass()));
            sb.append(String.format("| Image Size | %dx%d |\n", info.getImageWidth(), info.getImageHeight()));
            sb.append(String.format("| Train/Test | %d/%d |\n", info.getTrainSize(), info.getTestSize()));
            sb.append("\n");
        });

        // Classification Metrics
        sb.append("## Classification Metrics\n\n");
        sb.append("| Metric | Value |\n");
        sb.append("|--------|-------|\n");
        sb.append(String.format("| Accuracy | %.4f (%.2f%%) |\n", result.getAccuracy(), result.getAccuracyPercent()));
        sb.append(String.format("| Precision | %.4f |\n", result.getPrecision()));
        sb.append(String.format("| Recall | %.4f |\n", result.getRecall()));
        sb.append(String.format("| F1-Score | %.4f |\n", result.getF1Score()));
        sb.append(String.format("| Specificity | %.4f |\n", result.getSpecificity()));
        sb.append("\n");

        // Error Rates
        sb.append("## Error Rates\n\n");
        sb.append("| Metric | Value |\n");
        sb.append("|--------|-------|\n");
        sb.append(String.format("| FAR (False Accept Rate) | %.4f (%.2f%%) |\n",
            result.getFalseAcceptRate(), result.getFalseAcceptRate() * 100));
        sb.append(String.format("| FRR (False Reject Rate) | %.4f (%.2f%%) |\n",
            result.getFalseRejectRate(), result.getFalseRejectRate() * 100));
        sb.append(String.format("| EER (Equal Error Rate) | %.4f (%.2f%%) |\n",
            result.getEqualErrorRate(), result.getEerPercent()));
        sb.append(String.format("| AUC-ROC | %.4f |\n", result.getAreaUnderCurve()));
        sb.append("\n");

        // Performance Metrics
        result.getPerformanceMetrics().ifPresent(metrics -> {
            sb.append("## Performance Metrics\n\n");
            sb.append("| Metric | Value |\n");
            sb.append("|--------|-------|\n");
            sb.append(String.format("| Throughput | %.2f samples/sec |\n", metrics.getThroughput()));
            sb.append(String.format("| Samples Processed | %d |\n", metrics.getTotalSamplesProcessed()));
            sb.append(String.format("| Peak Memory | %.2f MB |\n", metrics.getPeakMemoryMB()));

            if (metrics.getTotalTime() != null) {
                BenchmarkResult.TimingStats t = metrics.getTotalTime();
                sb.append("\n### Timing Statistics (ms)\n\n");
                sb.append("| Statistic | Total | Extraction | Matching |\n");
                sb.append("|-----------|-------|------------|----------|\n");
                sb.append(String.format("| Mean | %.2f | %.2f | %.2f |\n",
                    t.getMean(),
                    metrics.getExtractionTime() != null ? metrics.getExtractionTime().getMean() : 0,
                    metrics.getMatchingTime() != null ? metrics.getMatchingTime().getMean() : 0));
                sb.append(String.format("| Std Dev | %.2f | %.2f | %.2f |\n",
                    t.getStdDev(),
                    metrics.getExtractionTime() != null ? metrics.getExtractionTime().getStdDev() : 0,
                    metrics.getMatchingTime() != null ? metrics.getMatchingTime().getStdDev() : 0));
                sb.append(String.format("| P95 | %.2f | %.2f | %.2f |\n",
                    t.getP95(),
                    metrics.getExtractionTime() != null ? metrics.getExtractionTime().getP95() : 0,
                    metrics.getMatchingTime() != null ? metrics.getMatchingTime().getP95() : 0));
            }
            sb.append("\n");
        });

        // Confusion Matrix
        result.getConfusionMatrix().ifPresent(cm -> {
            sb.append("## Confusion Matrix\n\n");
            sb.append("```\n");
            sb.append(cm.toFormattedString());
            sb.append("```\n\n");
        });

        // Per-Class Metrics
        if (!result.getPerClassMetrics().isEmpty()) {
            sb.append("## Per-Class Metrics\n\n");
            sb.append("| Class | Precision | Recall | F1-Score | Samples |\n");
            sb.append("|-------|-----------|--------|----------|--------|\n");

            result.getPerClassMetrics().values().stream()
                .sorted(Comparator.comparing(BenchmarkResult.ClassMetrics::getClassName))
                .forEach(cm -> {
                    sb.append(String.format("| %s | %.4f | %.4f | %.4f | %d |\n",
                        cm.getClassName(), cm.getPrecision(), cm.getRecall(),
                        cm.getF1Score(), cm.getSamples()));
                });
            sb.append("\n");
        }

        // Configuration
        if (!result.getConfiguration().isEmpty()) {
            sb.append("## Configuration\n\n");
            sb.append("| Parameter | Value |\n");
            sb.append("|-----------|-------|\n");
            result.getConfiguration().forEach((key, value) ->
                sb.append(String.format("| %s | %s |\n", key, value)));
            sb.append("\n");
        }

        // ROC Curve Data (if present, include chart placeholder)
        if (!result.getRocCurve().isEmpty()) {
            sb.append("## ROC Curve Data\n\n");
            sb.append("The ROC curve data is available in the JSON output for visualization.\n\n");
        }

        return sb.toString();
    }

    /**
     * Generates a comparison Markdown report for multiple results.
     *
     * @param results map of name to benchmark result
     * @return formatted Markdown string
     */
    public String generateComparisonMarkdown(Map<String, BenchmarkResult> results) {
        StringBuilder sb = new StringBuilder();

        sb.append("# Benchmark Comparison Report\n\n");
        sb.append("**Generated:** ").append(LocalDateTime.now().format(DATE_FORMAT)).append("\n\n");

        // Summary Table
        sb.append("## Summary\n\n");
        sb.append("| Algorithm | Accuracy | Precision | Recall | F1-Score | EER | Throughput |\n");
        sb.append("|-----------|----------|-----------|--------|----------|-----|------------|\n");

        for (Map.Entry<String, BenchmarkResult> entry : results.entrySet()) {
            BenchmarkResult r = entry.getValue();
            double throughput = r.getPerformanceMetrics()
                .map(BenchmarkResult.PerformanceMetrics::getThroughput)
                .orElse(0.0);

            sb.append(String.format("| %s | %.4f | %.4f | %.4f | %.4f | %.4f | %.2f/s |\n",
                entry.getKey(), r.getAccuracy(), r.getPrecision(), r.getRecall(),
                r.getF1Score(), r.getEqualErrorRate(), throughput));
        }
        sb.append("\n");

        // Detailed sections for each result
        sb.append("## Detailed Results\n\n");
        for (Map.Entry<String, BenchmarkResult> entry : results.entrySet()) {
            sb.append("### ").append(entry.getKey()).append("\n\n");
            sb.append(generateMetricsSummary(entry.getValue()));
            sb.append("\n---\n\n");
        }

        return sb.toString();
    }

    /**
     * Generates a brief metrics summary.
     */
    private String generateMetricsSummary(BenchmarkResult result) {
        StringBuilder sb = new StringBuilder();

        sb.append(String.format("- **Accuracy:** %.4f (%.2f%%)%n", result.getAccuracy(), result.getAccuracyPercent()));
        sb.append(String.format("- **Precision:** %.4f%n", result.getPrecision()));
        sb.append(String.format("- **Recall:** %.4f%n", result.getRecall()));
        sb.append(String.format("- **F1-Score:** %.4f%n", result.getF1Score()));
        sb.append(String.format("- **EER:** %.4f (%.2f%%)%n", result.getEqualErrorRate(), result.getEerPercent()));

        result.getPerformanceMetrics().ifPresent(metrics -> {
            sb.append(String.format("- **Throughput:** %.2f samples/sec%n", metrics.getThroughput()));
            if (metrics.getTotalTime() != null) {
                sb.append(String.format("- **Avg Processing Time:** %.2f ms%n", metrics.getTotalTime().getMean()));
            }
        });

        return sb.toString();
    }

    // ==================== JSON Generation ====================

    /**
     * Generates JSON representation of a benchmark result.
     *
     * @param result the benchmark result
     * @return JSON string
     * @throws IOException if serialization fails
     */
    public String generateJson(BenchmarkResult result) throws IOException {
        return objectMapper.writeValueAsString(resultToMap(result));
    }

    /**
     * Generates JSON for multiple results.
     *
     * @param results map of name to benchmark result
     * @return JSON string
     * @throws IOException if serialization fails
     */
    public String generateJson(Map<String, BenchmarkResult> results) throws IOException {
        Map<String, Object> output = new LinkedHashMap<>();
        output.put("generated", LocalDateTime.now().format(DATE_FORMAT));
        output.put("count", results.size());

        List<Map<String, Object>> resultList = new ArrayList<>();
        for (Map.Entry<String, BenchmarkResult> entry : results.entrySet()) {
            Map<String, Object> resultMap = resultToMap(entry.getValue());
            resultMap.put("label", entry.getKey());
            resultList.add(resultMap);
        }
        output.put("results", resultList);

        return objectMapper.writeValueAsString(output);
    }

    /**
     * Converts a BenchmarkResult to a Map for JSON serialization.
     */
    private Map<String, Object> resultToMap(BenchmarkResult result) {
        Map<String, Object> map = new LinkedHashMap<>();

        // Basic info
        map.put("id", result.getId());
        map.put("name", result.getName());
        map.put("description", result.getDescription());
        map.put("algorithmName", result.getAlgorithmName());
        map.put("timestamp", result.getTimestamp().format(DATE_FORMAT));

        // Classification metrics
        Map<String, Object> metrics = new LinkedHashMap<>();
        metrics.put("accuracy", result.getAccuracy());
        metrics.put("precision", result.getPrecision());
        metrics.put("recall", result.getRecall());
        metrics.put("f1Score", result.getF1Score());
        metrics.put("specificity", result.getSpecificity());
        map.put("classificationMetrics", metrics);

        // Error rates
        Map<String, Object> errorRates = new LinkedHashMap<>();
        errorRates.put("falseAcceptRate", result.getFalseAcceptRate());
        errorRates.put("falseRejectRate", result.getFalseRejectRate());
        errorRates.put("equalErrorRate", result.getEqualErrorRate());
        errorRates.put("areaUnderCurve", result.getAreaUnderCurve());
        map.put("errorRates", errorRates);

        // ROC curve
        if (!result.getRocCurve().isEmpty()) {
            List<Map<String, Double>> rocData = new ArrayList<>();
            for (BenchmarkResult.RocPoint point : result.getRocCurve()) {
                Map<String, Double> p = new LinkedHashMap<>();
                p.put("threshold", point.getThreshold());
                p.put("tpr", point.getTruePositiveRate());
                p.put("fpr", point.getFalsePositiveRate());
                rocData.add(p);
            }
            map.put("rocCurve", rocData);
        }

        // DET curve
        if (!result.getDetCurve().isEmpty()) {
            List<Map<String, Double>> detData = new ArrayList<>();
            for (BenchmarkResult.DetPoint point : result.getDetCurve()) {
                Map<String, Double> p = new LinkedHashMap<>();
                p.put("threshold", point.getThreshold());
                p.put("frr", point.getFalseRejectRate());
                p.put("far", point.getFalseAcceptRate());
                detData.add(p);
            }
            map.put("detCurve", detData);
        }

        // Performance metrics
        result.getPerformanceMetrics().ifPresent(perf -> {
            Map<String, Object> perfMap = new LinkedHashMap<>();
            perfMap.put("throughput", perf.getThroughput());
            perfMap.put("totalSamples", perf.getTotalSamplesProcessed());
            perfMap.put("peakMemoryBytes", perf.getPeakMemoryUsage());
            perfMap.put("avgMemoryBytes", perf.getAverageMemoryUsage());

            if (perf.getTotalTime() != null) {
                perfMap.put("totalTime", timingStatsToMap(perf.getTotalTime()));
            }
            if (perf.getExtractionTime() != null) {
                perfMap.put("extractionTime", timingStatsToMap(perf.getExtractionTime()));
            }
            if (perf.getMatchingTime() != null) {
                perfMap.put("matchingTime", timingStatsToMap(perf.getMatchingTime()));
            }

            map.put("performanceMetrics", perfMap);
        });

        // Confusion matrix
        result.getConfusionMatrix().ifPresent(cm -> {
            Map<String, Object> cmMap = new LinkedHashMap<>();
            cmMap.put("labels", cm.getLabels());
            cmMap.put("matrix", cm.getMatrix());
            cmMap.put("totalSamples", cm.getTotalSamples());
            map.put("confusionMatrix", cmMap);
        });

        // Per-class metrics
        if (!result.getPerClassMetrics().isEmpty()) {
            List<Map<String, Object>> classMetrics = new ArrayList<>();
            for (BenchmarkResult.ClassMetrics cm : result.getPerClassMetrics().values()) {
                Map<String, Object> classMap = new LinkedHashMap<>();
                classMap.put("className", cm.getClassName());
                classMap.put("samples", cm.getSamples());
                classMap.put("precision", cm.getPrecision());
                classMap.put("recall", cm.getRecall());
                classMap.put("f1Score", cm.getF1Score());
                classMap.put("specificity", cm.getSpecificity());
                classMetrics.add(classMap);
            }
            map.put("perClassMetrics", classMetrics);
        }

        // Dataset info
        result.getDatasetInfo().ifPresent(info -> {
            Map<String, Object> datasetMap = new LinkedHashMap<>();
            datasetMap.put("name", info.getName());
            datasetMap.put("format", info.getFormat());
            datasetMap.put("totalImages", info.getTotalImages());
            datasetMap.put("numClasses", info.getNumClasses());
            datasetMap.put("imagesPerClass", info.getImagesPerClass());
            datasetMap.put("imageWidth", info.getImageWidth());
            datasetMap.put("imageHeight", info.getImageHeight());
            datasetMap.put("trainSize", info.getTrainSize());
            datasetMap.put("testSize", info.getTestSize());
            map.put("datasetInfo", datasetMap);
        });

        // Configuration
        if (!result.getConfiguration().isEmpty()) {
            map.put("configuration", result.getConfiguration());
        }

        return map;
    }

    private Map<String, Double> timingStatsToMap(BenchmarkResult.TimingStats stats) {
        Map<String, Double> map = new LinkedHashMap<>();
        map.put("min", stats.getMin());
        map.put("max", stats.getMax());
        map.put("mean", stats.getMean());
        map.put("median", stats.getMedian());
        map.put("stdDev", stats.getStdDev());
        map.put("p95", stats.getP95());
        map.put("p99", stats.getP99());
        return map;
    }

    // ==================== HTML Generation ====================

    /**
     * Generates an HTML report with interactive charts.
     *
     * @param result the benchmark result
     * @return HTML string
     */
    public String generateHtml(BenchmarkResult result) {
        return generateHtml(Map.of(result.getName(), result));
    }

    /**
     * Generates an HTML report for multiple results with charts.
     *
     * @param results map of name to benchmark result
     * @return HTML string
     */
    public String generateHtml(Map<String, BenchmarkResult> results) {
        StringBuilder sb = new StringBuilder();

        // HTML Header
        sb.append("<!DOCTYPE html>\n");
        sb.append("<html lang=\"en\">\n");
        sb.append("<head>\n");
        sb.append("  <meta charset=\"UTF-8\">\n");
        sb.append("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n");
        sb.append("  <title>Face Recognition Benchmark Report</title>\n");
        sb.append("  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>\n");
        sb.append(generateHtmlStyles());
        sb.append("</head>\n");
        sb.append("<body>\n");

        // Header Section
        sb.append("<div class=\"container\">\n");
        sb.append("  <h1>Face Recognition Benchmark Report</h1>\n");
        sb.append("  <p class=\"timestamp\">Generated: ")
            .append(LocalDateTime.now().format(DATE_FORMAT)).append("</p>\n\n");

        // Summary Table
        sb.append("  <h2>Summary</h2>\n");
        sb.append("  <table class=\"summary-table\">\n");
        sb.append("    <thead>\n");
        sb.append("      <tr><th>Algorithm</th><th>Accuracy</th><th>Precision</th>");
        sb.append("<th>Recall</th><th>F1-Score</th><th>EER</th><th>Throughput</th></tr>\n");
        sb.append("    </thead>\n");
        sb.append("    <tbody>\n");

        for (Map.Entry<String, BenchmarkResult> entry : results.entrySet()) {
            BenchmarkResult r = entry.getValue();
            double throughput = r.getPerformanceMetrics()
                .map(BenchmarkResult.PerformanceMetrics::getThroughput)
                .orElse(0.0);

            sb.append(String.format("      <tr><td>%s</td><td>%.4f</td><td>%.4f</td><td>%.4f</td>" +
                    "<td>%.4f</td><td>%.4f</td><td>%.2f/s</td></tr>\n",
                entry.getKey(), r.getAccuracy(), r.getPrecision(), r.getRecall(),
                r.getF1Score(), r.getEqualErrorRate(), throughput));
        }
        sb.append("    </tbody>\n");
        sb.append("  </table>\n\n");

        // Charts Section
        sb.append("  <h2>Visualizations</h2>\n");

        // Accuracy Comparison Chart
        sb.append("  <div class=\"chart-container\">\n");
        sb.append("    <h3>Accuracy Comparison</h3>\n");
        sb.append("    <canvas id=\"accuracyChart\"></canvas>\n");
        sb.append("  </div>\n");

        // ROC Curves (if available)
        boolean hasRocData = results.values().stream().anyMatch(r -> !r.getRocCurve().isEmpty());
        if (hasRocData) {
            sb.append("  <div class=\"chart-container\">\n");
            sb.append("    <h3>ROC Curves</h3>\n");
            sb.append("    <canvas id=\"rocChart\"></canvas>\n");
            sb.append("  </div>\n");
        }

        // Detailed Results
        sb.append("  <h2>Detailed Results</h2>\n");
        for (Map.Entry<String, BenchmarkResult> entry : results.entrySet()) {
            sb.append(generateHtmlResultSection(entry.getKey(), entry.getValue()));
        }

        sb.append("</div>\n");

        // JavaScript for Charts
        sb.append("<script>\n");
        sb.append(generateChartScript(results));
        sb.append("</script>\n");

        sb.append("</body>\n");
        sb.append("</html>\n");

        return sb.toString();
    }

    private String generateHtmlStyles() {
        return "<style>\n" +
            "  body {\n" +
            "    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;\n" +
            "    line-height: 1.6;\n" +
            "    color: #333;\n" +
            "    background: #f5f5f5;\n" +
            "    margin: 0;\n" +
            "    padding: 0;\n" +
            "  }\n" +
            "  .container {\n" +
            "    max-width: 1200px;\n" +
            "    margin: 0 auto;\n" +
            "    padding: 20px;\n" +
            "    background: white;\n" +
            "    box-shadow: 0 2px 5px rgba(0,0,0,0.1);\n" +
            "  }\n" +
            "  h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }\n" +
            "  h2 { color: #34495e; margin-top: 30px; }\n" +
            "  h3 { color: #7f8c8d; }\n" +
            "  .timestamp { color: #7f8c8d; font-style: italic; }\n" +
            "  .summary-table {\n" +
            "    width: 100%;\n" +
            "    border-collapse: collapse;\n" +
            "    margin: 20px 0;\n" +
            "  }\n" +
            "  .summary-table th, .summary-table td {\n" +
            "    padding: 12px;\n" +
            "    text-align: left;\n" +
            "    border-bottom: 1px solid #ddd;\n" +
            "  }\n" +
            "  .summary-table th {\n" +
            "    background: #3498db;\n" +
            "    color: white;\n" +
            "  }\n" +
            "  .summary-table tr:hover { background: #f5f5f5; }\n" +
            "  .chart-container {\n" +
            "    margin: 30px 0;\n" +
            "    padding: 20px;\n" +
            "    background: #fafafa;\n" +
            "    border-radius: 8px;\n" +
            "  }\n" +
            "  .chart-container canvas {\n" +
            "    max-height: 400px;\n" +
            "  }\n" +
            "  .result-section {\n" +
            "    margin: 20px 0;\n" +
            "    padding: 20px;\n" +
            "    border: 1px solid #ddd;\n" +
            "    border-radius: 8px;\n" +
            "    background: #fafafa;\n" +
            "  }\n" +
            "  .metrics-grid {\n" +
            "    display: grid;\n" +
            "    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));\n" +
            "    gap: 15px;\n" +
            "    margin: 15px 0;\n" +
            "  }\n" +
            "  .metric-card {\n" +
            "    padding: 15px;\n" +
            "    background: white;\n" +
            "    border-radius: 5px;\n" +
            "    box-shadow: 0 1px 3px rgba(0,0,0,0.1);\n" +
            "  }\n" +
            "  .metric-card .label { font-size: 0.85em; color: #7f8c8d; }\n" +
            "  .metric-card .value { font-size: 1.5em; font-weight: bold; color: #2c3e50; }\n" +
            "  .confusion-matrix {\n" +
            "    overflow-x: auto;\n" +
            "    margin: 15px 0;\n" +
            "  }\n" +
            "  .confusion-matrix table {\n" +
            "    border-collapse: collapse;\n" +
            "    font-size: 0.9em;\n" +
            "  }\n" +
            "  .confusion-matrix th, .confusion-matrix td {\n" +
            "    padding: 8px;\n" +
            "    border: 1px solid #ddd;\n" +
            "    text-align: center;\n" +
            "  }\n" +
            "  .confusion-matrix th { background: #ecf0f1; }\n" +
            "</style>\n";
    }

    private String generateHtmlResultSection(String name, BenchmarkResult result) {
        StringBuilder sb = new StringBuilder();

        sb.append("  <div class=\"result-section\">\n");
        sb.append("    <h3>").append(name).append("</h3>\n");

        // Metrics grid
        sb.append("    <div class=\"metrics-grid\">\n");
        sb.append(generateMetricCard("Accuracy", String.format("%.2f%%", result.getAccuracyPercent())));
        sb.append(generateMetricCard("Precision", String.format("%.4f", result.getPrecision())));
        sb.append(generateMetricCard("Recall", String.format("%.4f", result.getRecall())));
        sb.append(generateMetricCard("F1-Score", String.format("%.4f", result.getF1Score())));
        sb.append(generateMetricCard("EER", String.format("%.2f%%", result.getEerPercent())));
        sb.append(generateMetricCard("AUC", String.format("%.4f", result.getAreaUnderCurve())));
        sb.append("    </div>\n");

        // Performance metrics
        result.getPerformanceMetrics().ifPresent(metrics -> {
            sb.append("    <div class=\"metrics-grid\">\n");
            sb.append(generateMetricCard("Throughput", String.format("%.2f/s", metrics.getThroughput())));
            if (metrics.getTotalTime() != null) {
                sb.append(generateMetricCard("Avg Time", String.format("%.2fms", metrics.getTotalTime().getMean())));
            }
            sb.append(generateMetricCard("Peak Memory", String.format("%.2fMB", metrics.getPeakMemoryMB())));
            sb.append("    </div>\n");
        });

        // Confusion matrix
        result.getConfusionMatrix().ifPresent(cm -> {
            sb.append("    <details>\n");
            sb.append("      <summary>Confusion Matrix</summary>\n");
            sb.append("      <div class=\"confusion-matrix\">\n");
            sb.append("        <table>\n");
            sb.append("          <tr><th></th>");
            for (String label : cm.getLabels()) {
                sb.append("<th>").append(truncate(label, 10)).append("</th>");
            }
            sb.append("</tr>\n");

            int[][] matrix = cm.getMatrix();
            for (int i = 0; i < matrix.length; i++) {
                sb.append("          <tr><th>").append(truncate(cm.getLabels().get(i), 10)).append("</th>");
                for (int j = 0; j < matrix[i].length; j++) {
                    String style = i == j ? " style=\"background:#d4edda;\"" : "";
                    sb.append("<td").append(style).append(">").append(matrix[i][j]).append("</td>");
                }
                sb.append("</tr>\n");
            }
            sb.append("        </table>\n");
            sb.append("      </div>\n");
            sb.append("    </details>\n");
        });

        sb.append("  </div>\n");
        return sb.toString();
    }

    private String generateMetricCard(String label, String value) {
        return String.format(
            "      <div class=\"metric-card\">\n" +
            "        <div class=\"label\">%s</div>\n" +
            "        <div class=\"value\">%s</div>\n" +
            "      </div>\n", label, value);
    }

    private String generateChartScript(Map<String, BenchmarkResult> results) {
        StringBuilder sb = new StringBuilder();

        // Accuracy comparison chart
        sb.append("// Accuracy Comparison Chart\n");
        sb.append("new Chart(document.getElementById('accuracyChart'), {\n");
        sb.append("  type: 'bar',\n");
        sb.append("  data: {\n");
        sb.append("    labels: [");
        sb.append(results.keySet().stream().map(s -> "'" + s + "'").reduce((a, b) -> a + "," + b).orElse(""));
        sb.append("],\n");
        sb.append("    datasets: [{\n");
        sb.append("      label: 'Accuracy',\n");
        sb.append("      data: [");
        sb.append(results.values().stream().map(r -> String.format("%.4f", r.getAccuracy()))
            .reduce((a, b) -> a + "," + b).orElse(""));
        sb.append("],\n");
        sb.append("      backgroundColor: 'rgba(52, 152, 219, 0.7)',\n");
        sb.append("      borderColor: 'rgba(52, 152, 219, 1)',\n");
        sb.append("      borderWidth: 1\n");
        sb.append("    }, {\n");
        sb.append("      label: 'F1-Score',\n");
        sb.append("      data: [");
        sb.append(results.values().stream().map(r -> String.format("%.4f", r.getF1Score()))
            .reduce((a, b) -> a + "," + b).orElse(""));
        sb.append("],\n");
        sb.append("      backgroundColor: 'rgba(46, 204, 113, 0.7)',\n");
        sb.append("      borderColor: 'rgba(46, 204, 113, 1)',\n");
        sb.append("      borderWidth: 1\n");
        sb.append("    }]\n");
        sb.append("  },\n");
        sb.append("  options: {\n");
        sb.append("    responsive: true,\n");
        sb.append("    scales: { y: { beginAtZero: true, max: 1 } }\n");
        sb.append("  }\n");
        sb.append("});\n\n");

        // ROC curves
        boolean hasRocData = results.values().stream().anyMatch(r -> !r.getRocCurve().isEmpty());
        if (hasRocData) {
            sb.append("// ROC Curves\n");
            sb.append("new Chart(document.getElementById('rocChart'), {\n");
            sb.append("  type: 'line',\n");
            sb.append("  data: {\n");
            sb.append("    datasets: [\n");

            String[] colors = {"rgba(231, 76, 60, 1)", "rgba(52, 152, 219, 1)",
                "rgba(46, 204, 113, 1)", "rgba(155, 89, 182, 1)", "rgba(241, 196, 15, 1)"};
            int colorIdx = 0;

            for (Map.Entry<String, BenchmarkResult> entry : results.entrySet()) {
                if (!entry.getValue().getRocCurve().isEmpty()) {
                    sb.append("      {\n");
                    sb.append("        label: '").append(entry.getKey()).append("',\n");
                    sb.append("        data: [");

                    List<BenchmarkResult.RocPoint> roc = entry.getValue().getRocCurve();
                    List<BenchmarkResult.RocPoint> sorted = new ArrayList<>(roc);
                    sorted.sort(Comparator.comparingDouble(BenchmarkResult.RocPoint::getFalsePositiveRate));

                    for (int i = 0; i < sorted.size(); i++) {
                        BenchmarkResult.RocPoint p = sorted.get(i);
                        if (i > 0) sb.append(",");
                        sb.append(String.format("{x:%.4f,y:%.4f}", p.getFalsePositiveRate(), p.getTruePositiveRate()));
                    }
                    sb.append("],\n");
                    sb.append("        borderColor: '").append(colors[colorIdx % colors.length]).append("',\n");
                    sb.append("        fill: false,\n");
                    sb.append("        tension: 0.1\n");
                    sb.append("      },\n");
                    colorIdx++;
                }
            }

            // Diagonal line
            sb.append("      {\n");
            sb.append("        label: 'Random',\n");
            sb.append("        data: [{x:0,y:0},{x:1,y:1}],\n");
            sb.append("        borderColor: 'rgba(149, 165, 166, 0.5)',\n");
            sb.append("        borderDash: [5,5],\n");
            sb.append("        fill: false\n");
            sb.append("      }\n");

            sb.append("    ]\n");
            sb.append("  },\n");
            sb.append("  options: {\n");
            sb.append("    responsive: true,\n");
            sb.append("    scales: {\n");
            sb.append("      x: { type: 'linear', title: { display: true, text: 'False Positive Rate' }, min: 0, max: 1 },\n");
            sb.append("      y: { type: 'linear', title: { display: true, text: 'True Positive Rate' }, min: 0, max: 1 }\n");
            sb.append("    }\n");
            sb.append("  }\n");
            sb.append("});\n");
        }

        return sb.toString();
    }

    private static String truncate(String s, int maxLen) {
        return s.length() <= maxLen ? s : s.substring(0, maxLen - 2) + "..";
    }

    // ==================== CSV Generation ====================

    /**
     * Generates a CSV summary of benchmark results.
     *
     * @param results map of name to benchmark result
     * @return CSV string
     */
    public String generateCsv(Map<String, BenchmarkResult> results) {
        StringBuilder sb = new StringBuilder();

        // Header
        sb.append("Name,Algorithm,Accuracy,Precision,Recall,F1-Score,Specificity,");
        sb.append("FAR,FRR,EER,AUC,Throughput,AvgTimeMs,PeakMemoryMB,Timestamp\n");

        // Data rows
        for (Map.Entry<String, BenchmarkResult> entry : results.entrySet()) {
            BenchmarkResult r = entry.getValue();

            double throughput = r.getPerformanceMetrics()
                .map(BenchmarkResult.PerformanceMetrics::getThroughput).orElse(0.0);
            double avgTime = r.getPerformanceMetrics()
                .flatMap(m -> Optional.ofNullable(m.getTotalTime()))
                .map(BenchmarkResult.TimingStats::getMean).orElse(0.0);
            double peakMem = r.getPerformanceMetrics()
                .map(BenchmarkResult.PerformanceMetrics::getPeakMemoryMB).orElse(0.0);

            sb.append(String.format("\"%s\",\"%s\",%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.2f,%.4f,%.2f,\"%s\"\n",
                entry.getKey(),
                r.getAlgorithmName(),
                r.getAccuracy(),
                r.getPrecision(),
                r.getRecall(),
                r.getF1Score(),
                r.getSpecificity(),
                r.getFalseAcceptRate(),
                r.getFalseRejectRate(),
                r.getEqualErrorRate(),
                r.getAreaUnderCurve(),
                throughput,
                avgTime,
                peakMem,
                r.getTimestamp().format(DATE_FORMAT)));
        }

        return sb.toString();
    }

    // ==================== File Saving ====================

    /**
     * Saves a Markdown report to file.
     *
     * @param result the benchmark result
     * @param path output file path
     * @throws IOException if writing fails
     */
    public void saveMarkdown(BenchmarkResult result, Path path) throws IOException {
        Files.writeString(path, generateMarkdown(result));
    }

    /**
     * Saves a comparison Markdown report to file.
     *
     * @param results map of results
     * @param path output file path
     * @throws IOException if writing fails
     */
    public void saveMarkdown(Map<String, BenchmarkResult> results, Path path) throws IOException {
        Files.writeString(path, generateComparisonMarkdown(results));
    }

    /**
     * Saves a JSON report to file.
     *
     * @param result the benchmark result
     * @param path output file path
     * @throws IOException if writing fails
     */
    public void saveJson(BenchmarkResult result, Path path) throws IOException {
        Files.writeString(path, generateJson(result));
    }

    /**
     * Saves a JSON report for multiple results to file.
     *
     * @param results map of results
     * @param path output file path
     * @throws IOException if writing fails
     */
    public void saveJson(Map<String, BenchmarkResult> results, Path path) throws IOException {
        Files.writeString(path, generateJson(results));
    }

    /**
     * Saves an HTML report to file.
     *
     * @param result the benchmark result
     * @param path output file path
     * @throws IOException if writing fails
     */
    public void saveHtml(BenchmarkResult result, Path path) throws IOException {
        Files.writeString(path, generateHtml(result));
    }

    /**
     * Saves an HTML report for multiple results to file.
     *
     * @param results map of results
     * @param path output file path
     * @throws IOException if writing fails
     */
    public void saveHtml(Map<String, BenchmarkResult> results, Path path) throws IOException {
        Files.writeString(path, generateHtml(results));
    }

    /**
     * Saves a CSV report to file.
     *
     * @param results map of results
     * @param path output file path
     * @throws IOException if writing fails
     */
    public void saveCsv(Map<String, BenchmarkResult> results, Path path) throws IOException {
        Files.writeString(path, generateCsv(results));
    }

    /**
     * Saves all report formats to a directory.
     *
     * @param results map of results
     * @param outputDir output directory
     * @param baseName base filename (without extension)
     * @throws IOException if writing fails
     */
    public void saveAllFormats(Map<String, BenchmarkResult> results, Path outputDir, String baseName)
            throws IOException {
        Files.createDirectories(outputDir);

        saveMarkdown(results, outputDir.resolve(baseName + ".md"));
        saveJson(results, outputDir.resolve(baseName + ".json"));
        saveHtml(results, outputDir.resolve(baseName + ".html"));
        saveCsv(results, outputDir.resolve(baseName + ".csv"));
    }
}
