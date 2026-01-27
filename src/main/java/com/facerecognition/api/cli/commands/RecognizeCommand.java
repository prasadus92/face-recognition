package com.facerecognition.api.cli.commands;

import com.facerecognition.api.cli.FaceRecognitionCLI;
import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;
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
import java.util.*;
import java.util.concurrent.Callable;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * CLI command for recognizing faces in images.
 *
 * <p>This command supports both single image recognition and batch
 * processing of directories containing multiple images.</p>
 *
 * <h3>Examples:</h3>
 * <pre>
 * # Recognize a single image
 * face-recognition recognize -i photo.jpg --model model.dat
 *
 * # Recognize all images in a directory
 * face-recognition recognize -i photos/ --model model.dat --recursive
 *
 * # Output results as JSON
 * face-recognition recognize -i photo.jpg --model model.dat --output json
 *
 * # Use custom threshold
 * face-recognition recognize -i photo.jpg --model model.dat --threshold 0.8
 * </pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Command(
    name = "recognize",
    aliases = {"rec", "identify"},
    description = "Recognize faces in images using a trained model.",
    mixinStandardHelpOptions = true,
    sortOptions = false,
    footer = {
        "",
        "Output Formats:",
        "  text   Human-readable text output (default)",
        "  json   JSON format suitable for parsing",
        "  csv    CSV format for spreadsheet import",
        "",
        "Examples:",
        "  @|bold face-recognition recognize -i photo.jpg --model model.dat|@",
        "  @|bold face-recognition recognize -i photos/ --recursive --output json|@"
    }
)
public class RecognizeCommand implements Callable<Integer> {

    @ParentCommand
    private FaceRecognitionCLI parent;

    @Parameters(index = "0", arity = "0..1",
            description = "Image file or directory to recognize (alternative to -i)")
    private File inputParam;

    @Option(names = {"-i", "--input"}, paramLabel = "PATH",
            description = "Image file or directory to recognize")
    private File inputFile;

    @Option(names = {"-m", "--model"}, paramLabel = "FILE",
            description = "Path to the trained model file",
            required = true)
    private File modelFile;

    @Option(names = {"-t", "--threshold"}, paramLabel = "VALUE",
            description = "Recognition confidence threshold (0.0-1.0, default: ${DEFAULT-VALUE})",
            defaultValue = "0.6")
    private double threshold;

    @Option(names = {"-o", "--output"}, paramLabel = "FORMAT",
            description = "Output format: ${COMPLETION-CANDIDATES} (default: ${DEFAULT-VALUE})",
            defaultValue = "text")
    private OutputFormat outputFormat;

    @Option(names = {"--output-file"}, paramLabel = "FILE",
            description = "Write results to file instead of stdout")
    private File outputFile;

    @Option(names = {"-r", "--recursive"},
            description = "Process directories recursively")
    private boolean recursive;

    @Option(names = {"-v", "--verbose"},
            description = "Show detailed recognition information")
    private boolean verbose;

    @Option(names = {"--top-n"}, paramLabel = "N",
            description = "Show top N matches (default: ${DEFAULT-VALUE})",
            defaultValue = "3")
    private int topN;

    @Option(names = {"--min-confidence"}, paramLabel = "VALUE",
            description = "Minimum confidence to display alternative matches (default: ${DEFAULT-VALUE})",
            defaultValue = "0.3")
    private double minConfidence;

    @Option(names = {"--extensions"}, paramLabel = "EXT",
            description = "File extensions to process (default: jpg,jpeg,png,bmp)",
            split = ",",
            defaultValue = "jpg,jpeg,png,bmp")
    private Set<String> extensions;

    @Option(names = {"--parallel"},
            description = "Process images in parallel (faster for large batches)")
    private boolean parallel;

    @Option(names = {"--show-timing"},
            description = "Show processing time for each image")
    private boolean showTiming;

    /**
     * Output format options.
     */
    public enum OutputFormat {
        text, json, csv
    }

    @Override
    public Integer call() throws Exception {
        // Determine input source
        File input = inputFile != null ? inputFile : inputParam;
        if (input == null) {
            System.err.println("Error: No input file or directory specified. Use -i or provide as argument.");
            return FaceRecognitionCLI.EXIT_INVALID_ARGS;
        }

        if (!input.exists()) {
            System.err.println("Error: Input not found: " + input);
            return FaceRecognitionCLI.EXIT_FILE_NOT_FOUND;
        }

        if (!modelFile.exists()) {
            System.err.println("Error: Model file not found: " + modelFile);
            return FaceRecognitionCLI.EXIT_FILE_NOT_FOUND;
        }

        // Load the model
        FaceRecognitionService service = loadModel(modelFile);
        if (service == null) {
            return FaceRecognitionCLI.EXIT_ERROR;
        }

        if (!service.isTrained()) {
            System.err.println("Error: Model is not trained");
            return FaceRecognitionCLI.EXIT_MODEL_NOT_TRAINED;
        }

        // Collect files to process
        List<File> files = collectFiles(input);
        if (files.isEmpty()) {
            System.err.println("Error: No image files found to process");
            return FaceRecognitionCLI.EXIT_FILE_NOT_FOUND;
        }

        if (verbose && !isQuiet()) {
            System.out.println("Found " + files.size() + " image(s) to process");
            System.out.println("Using model: " + modelFile);
            System.out.println("Recognition threshold: " + threshold);
            System.out.println();
        }

        // Process files and collect results
        List<RecognitionOutput> results = processFiles(files, service);

        // Output results
        outputResults(results);

        // Summary
        if (!isQuiet() && files.size() > 1) {
            printSummary(results);
        }

        // Determine exit code
        boolean anyRecognized = results.stream()
            .anyMatch(r -> r.result != null && r.result.isRecognized());

        return anyRecognized ? FaceRecognitionCLI.EXIT_SUCCESS : FaceRecognitionCLI.EXIT_RECOGNITION_FAILED;
    }

    private boolean isQuiet() {
        return parent != null && parent.isQuiet();
    }

    private FaceRecognitionService loadModel(File modelFile) {
        try {
            if (verbose && !isQuiet()) {
                System.out.println("Loading model from: " + modelFile);
            }

            try (ObjectInputStream ois = new ObjectInputStream(
                    new BufferedInputStream(new FileInputStream(modelFile)))) {
                Object obj = ois.readObject();

                if (obj instanceof FaceRecognitionService) {
                    return (FaceRecognitionService) obj;
                } else if (obj instanceof FeatureExtractor) {
                    // Create service from extractor
                    FeatureExtractor extractor = (FeatureExtractor) obj;
                    return FaceRecognitionService.builder()
                        .extractor(extractor)
                        .classifier(new KNNClassifier())
                        .build();
                } else {
                    System.err.println("Error: Unknown model type: " + obj.getClass().getName());
                    return null;
                }
            }
        } catch (IOException | ClassNotFoundException e) {
            System.err.println("Error loading model: " + e.getMessage());
            return null;
        }
    }

    private List<File> collectFiles(File input) {
        List<File> files = new ArrayList<>();

        if (input.isFile()) {
            if (isImageFile(input)) {
                files.add(input);
            }
        } else if (input.isDirectory()) {
            try {
                int maxDepth = recursive ? Integer.MAX_VALUE : 1;
                try (Stream<Path> stream = Files.walk(input.toPath(), maxDepth)) {
                    files = stream
                        .map(Path::toFile)
                        .filter(File::isFile)
                        .filter(this::isImageFile)
                        .sorted()
                        .collect(Collectors.toList());
                }
            } catch (IOException e) {
                System.err.println("Error scanning directory: " + e.getMessage());
            }
        }

        return files;
    }

    private boolean isImageFile(File file) {
        String name = file.getName().toLowerCase();
        return extensions.stream().anyMatch(ext -> name.endsWith("." + ext.toLowerCase()));
    }

    private List<RecognitionOutput> processFiles(List<File> files, FaceRecognitionService service) {
        List<RecognitionOutput> results = new ArrayList<>();
        int total = files.size();
        int current = 0;

        Stream<File> stream = parallel ? files.parallelStream() : files.stream();

        if (parallel) {
            results = stream.map(file -> processFile(file, service)).collect(Collectors.toList());
        } else {
            for (File file : files) {
                current++;
                if (!isQuiet() && total > 1 && outputFormat == OutputFormat.text) {
                    FaceRecognitionCLI.printProgressBar(current, total, 40);
                }
                results.add(processFile(file, service));
            }
        }

        return results;
    }

    private RecognitionOutput processFile(File file, FaceRecognitionService service) {
        RecognitionOutput output = new RecognitionOutput();
        output.file = file;
        output.startTime = System.currentTimeMillis();

        try {
            FaceImage image = FaceImage.fromFile(file);
            output.result = service.recognize(image);
            output.success = true;
        } catch (IOException e) {
            output.error = "Failed to read image: " + e.getMessage();
            output.success = false;
        } catch (Exception e) {
            output.error = "Recognition error: " + e.getMessage();
            output.success = false;
        }

        output.processingTime = System.currentTimeMillis() - output.startTime;
        return output;
    }

    private void outputResults(List<RecognitionOutput> results) throws IOException {
        PrintWriter writer;
        if (outputFile != null) {
            writer = new PrintWriter(new FileWriter(outputFile));
        } else {
            writer = new PrintWriter(System.out);
        }

        switch (outputFormat) {
            case json:
                outputJson(results, writer);
                break;
            case csv:
                outputCsv(results, writer);
                break;
            case text:
            default:
                outputText(results, writer);
                break;
        }

        if (outputFile != null) {
            writer.close();
            if (!isQuiet()) {
                System.out.println("\nResults written to: " + outputFile);
            }
        }
    }

    private void outputText(List<RecognitionOutput> results, PrintWriter writer) {
        for (RecognitionOutput output : results) {
            writer.println("=".repeat(60));
            writer.println("File: " + output.file.getName());

            if (!output.success) {
                writer.println("Status: ERROR - " + output.error);
                continue;
            }

            RecognitionResult result = output.result;
            writer.println("Status: " + result.getStatus());

            if (result.isRecognized()) {
                RecognitionResult.MatchResult match = result.getBestMatch().get();
                writer.println("Identity: " + match.getIdentity().getName());
                writer.printf("Confidence: %.2f%% (threshold: %.2f%%)%n",
                    match.getConfidence() * 100, threshold * 100);
                writer.printf("Distance: %.4f%n", match.getDistance());
            } else if (result.getStatus() == RecognitionResult.Status.UNKNOWN) {
                writer.println("Identity: [Unknown - no match above threshold]");
            } else if (result.getStatus() == RecognitionResult.Status.NO_FACE_DETECTED) {
                writer.println("No face detected in image");
            }

            // Show alternatives if verbose
            if (verbose && !result.getAlternatives().isEmpty()) {
                writer.println("\nAlternative Matches:");
                List<RecognitionResult.MatchResult> alternatives = result.getTopAlternatives(topN);
                for (int i = 0; i < alternatives.size(); i++) {
                    RecognitionResult.MatchResult alt = alternatives.get(i);
                    if (alt.getConfidence() >= minConfidence) {
                        writer.printf("  %d. %s (%.2f%%)%n",
                            i + 1, alt.getIdentity().getName(), alt.getConfidence() * 100);
                    }
                }
            }

            // Show timing
            if (showTiming || verbose) {
                writer.println("\nProcessing Time: " + FaceRecognitionCLI.formatDuration(output.processingTime));
                result.getMetrics().ifPresent(m -> {
                    writer.println("  Detection: " + m.getDetectionTimeMs() + "ms");
                    writer.println("  Extraction: " + m.getExtractionTimeMs() + "ms");
                    writer.println("  Matching: " + m.getMatchingTimeMs() + "ms");
                });
            }
        }
        writer.println("=".repeat(60));
    }

    private void outputJson(List<RecognitionOutput> results, PrintWriter writer) throws IOException {
        ObjectMapper mapper = new ObjectMapper();
        mapper.registerModule(new JavaTimeModule());
        mapper.enable(SerializationFeature.INDENT_OUTPUT);

        List<Map<String, Object>> jsonResults = new ArrayList<>();

        for (RecognitionOutput output : results) {
            Map<String, Object> item = new LinkedHashMap<>();
            item.put("file", output.file.getAbsolutePath());
            item.put("filename", output.file.getName());
            item.put("success", output.success);
            item.put("processingTimeMs", output.processingTime);

            if (!output.success) {
                item.put("error", output.error);
            } else {
                RecognitionResult result = output.result;
                item.put("status", result.getStatus().name());
                item.put("recognized", result.isRecognized());
                item.put("faceDetected", result.isFaceDetected());

                if (result.isRecognized()) {
                    RecognitionResult.MatchResult match = result.getBestMatch().get();
                    Map<String, Object> matchInfo = new LinkedHashMap<>();
                    matchInfo.put("identity", match.getIdentity().getName());
                    matchInfo.put("identityId", match.getIdentity().getId());
                    matchInfo.put("confidence", match.getConfidence());
                    matchInfo.put("distance", match.getDistance());
                    item.put("match", matchInfo);
                }

                if (!result.getAlternatives().isEmpty()) {
                    List<Map<String, Object>> altList = new ArrayList<>();
                    for (RecognitionResult.MatchResult alt : result.getTopAlternatives(topN)) {
                        if (alt.getConfidence() >= minConfidence) {
                            Map<String, Object> altInfo = new LinkedHashMap<>();
                            altInfo.put("identity", alt.getIdentity().getName());
                            altInfo.put("confidence", alt.getConfidence());
                            altInfo.put("distance", alt.getDistance());
                            altList.add(altInfo);
                        }
                    }
                    item.put("alternatives", altList);
                }

                result.getMetrics().ifPresent(m -> {
                    Map<String, Object> metrics = new LinkedHashMap<>();
                    metrics.put("totalMs", m.getTotalTimeMs());
                    metrics.put("detectionMs", m.getDetectionTimeMs());
                    metrics.put("extractionMs", m.getExtractionTimeMs());
                    metrics.put("matchingMs", m.getMatchingTimeMs());
                    item.put("metrics", metrics);
                });
            }

            jsonResults.add(item);
        }

        Map<String, Object> output = new LinkedHashMap<>();
        output.put("timestamp", java.time.LocalDateTime.now().toString());
        output.put("model", modelFile.getName());
        output.put("threshold", threshold);
        output.put("totalImages", results.size());
        output.put("results", jsonResults);

        writer.println(mapper.writeValueAsString(output));
    }

    private void outputCsv(List<RecognitionOutput> results, PrintWriter writer) {
        // Header
        writer.println("file,status,identity,confidence,distance,processing_time_ms,error");

        for (RecognitionOutput output : results) {
            StringBuilder line = new StringBuilder();
            line.append(escapeCsv(output.file.getName())).append(",");

            if (!output.success) {
                line.append("ERROR,,,,,");
                line.append(escapeCsv(output.error));
            } else {
                RecognitionResult result = output.result;
                line.append(result.getStatus().name()).append(",");

                if (result.isRecognized()) {
                    RecognitionResult.MatchResult match = result.getBestMatch().get();
                    line.append(escapeCsv(match.getIdentity().getName())).append(",");
                    line.append(String.format("%.4f", match.getConfidence())).append(",");
                    line.append(String.format("%.4f", match.getDistance())).append(",");
                } else {
                    line.append(",,,");
                }

                line.append(output.processingTime).append(",");
            }

            writer.println(line);
        }
    }

    private String escapeCsv(String value) {
        if (value == null) return "";
        if (value.contains(",") || value.contains("\"") || value.contains("\n")) {
            return "\"" + value.replace("\"", "\"\"") + "\"";
        }
        return value;
    }

    private void printSummary(List<RecognitionOutput> results) {
        System.out.println("\n" + "=".repeat(60));
        System.out.println("RECOGNITION SUMMARY");
        System.out.println("=".repeat(60));

        int total = results.size();
        long recognized = results.stream()
            .filter(r -> r.success && r.result != null && r.result.isRecognized())
            .count();
        long unknown = results.stream()
            .filter(r -> r.success && r.result != null &&
                    r.result.getStatus() == RecognitionResult.Status.UNKNOWN)
            .count();
        long noFace = results.stream()
            .filter(r -> r.success && r.result != null &&
                    r.result.getStatus() == RecognitionResult.Status.NO_FACE_DETECTED)
            .count();
        long errors = results.stream()
            .filter(r -> !r.success)
            .count();

        System.out.printf("Total Images:     %d%n", total);
        System.out.printf("Recognized:       %d (%.1f%%)%n", recognized, 100.0 * recognized / total);
        System.out.printf("Unknown:          %d (%.1f%%)%n", unknown, 100.0 * unknown / total);
        System.out.printf("No Face Detected: %d (%.1f%%)%n", noFace, 100.0 * noFace / total);
        System.out.printf("Errors:           %d (%.1f%%)%n", errors, 100.0 * errors / total);

        long totalTime = results.stream().mapToLong(r -> r.processingTime).sum();
        double avgTime = results.stream().mapToLong(r -> r.processingTime).average().orElse(0);
        System.out.printf("%nTotal Processing Time: %s%n", FaceRecognitionCLI.formatDuration(totalTime));
        System.out.printf("Average Time per Image: %.1fms%n", avgTime);

        // Identity distribution
        if (recognized > 0) {
            System.out.println("\nRecognized Identities:");
            Map<String, Long> identityCounts = results.stream()
                .filter(r -> r.success && r.result != null && r.result.isRecognized())
                .collect(Collectors.groupingBy(
                    r -> r.result.getBestMatch().get().getIdentity().getName(),
                    Collectors.counting()
                ));

            identityCounts.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .limit(10)
                .forEach(e -> System.out.printf("  %s: %d%n", e.getKey(), e.getValue()));
        }
    }

    /**
     * Container for recognition output data.
     */
    private static class RecognitionOutput {
        File file;
        RecognitionResult result;
        boolean success;
        String error;
        long startTime;
        long processingTime;
    }
}
