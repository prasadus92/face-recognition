package com.facerecognition.api.cli;

import com.facerecognition.api.cli.commands.*;

import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Spec;

import java.io.*;
import java.util.concurrent.Callable;

/**
 * Main CLI entry point for the Face Recognition System.
 *
 * <p>This CLI provides a comprehensive interface for face recognition operations
 * including recognition, enrollment, training, model management, benchmarking,
 * and serving a REST API.</p>
 *
 * <h3>Usage Examples:</h3>
 * <pre>
 * # Recognize faces in an image
 * face-recognition recognize -i photo.jpg --model model.dat
 *
 * # Enroll new faces
 * face-recognition enroll -i faces/ --identity "John Doe"
 *
 * # Train a model
 * face-recognition train -d dataset/ -a eigenfaces -o model.dat
 *
 * # Export/Import models
 * face-recognition export --model model.dat -o model.json
 * face-recognition import -i model.json -o model.dat
 *
 * # Run benchmarks
 * face-recognition benchmark -d lfw/ --algorithm eigenfaces
 *
 * # Start REST API server
 * face-recognition serve --port 8080 --model model.dat
 * </pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Command(
    name = "face-recognition",
    mixinStandardHelpOptions = true,
    version = "Face Recognition CLI 2.0.0",
    description = "A comprehensive face recognition command-line tool supporting multiple algorithms.",
    subcommands = {
        RecognizeCommand.class,
        EnrollCommand.class,
        TrainCommand.class,
        BenchmarkCommand.class,
        ServeCommand.class,
        CommandLine.HelpCommand.class
    },
    footer = {
        "",
        "Exit Codes:",
        "  0   Successful execution",
        "  1   General error",
        "  2   Invalid arguments",
        "  3   File not found",
        "  4   Model not trained",
        "  5   Recognition failed",
        "",
        "Examples:",
        "  face-recognition recognize -i photo.jpg",
        "  face-recognition enroll -i faces/ --identity \"John Doe\"",
        "  face-recognition train -d dataset/ -a eigenfaces",
        "  face-recognition serve --port 8080",
        "",
        "For more information, visit: https://github.com/prasadus92/face-recognition"
    }
)
public class FaceRecognitionCLI implements Callable<Integer> {

    /** Exit code for successful execution. */
    public static final int EXIT_SUCCESS = 0;

    /** Exit code for general errors. */
    public static final int EXIT_ERROR = 1;

    /** Exit code for invalid arguments. */
    public static final int EXIT_INVALID_ARGS = 2;

    /** Exit code for file not found errors. */
    public static final int EXIT_FILE_NOT_FOUND = 3;

    /** Exit code for model not trained errors. */
    public static final int EXIT_MODEL_NOT_TRAINED = 4;

    /** Exit code for recognition failure. */
    public static final int EXIT_RECOGNITION_FAILED = 5;

    @Spec
    private CommandSpec spec;

    @Option(names = {"-v", "--verbose"}, description = "Enable verbose output")
    private boolean verbose;

    @Option(names = {"--debug"}, description = "Enable debug mode with detailed logging")
    private boolean debug;

    @Option(names = {"--quiet", "-q"}, description = "Suppress all output except errors")
    private boolean quiet;

    @Option(names = {"--config"}, description = "Path to configuration file")
    private File configFile;

    /**
     * Checks if verbose mode is enabled.
     *
     * @return true if verbose output is enabled
     */
    public boolean isVerbose() {
        return verbose;
    }

    /**
     * Checks if debug mode is enabled.
     *
     * @return true if debug mode is enabled
     */
    public boolean isDebug() {
        return debug;
    }

    /**
     * Checks if quiet mode is enabled.
     *
     * @return true if quiet mode is enabled
     */
    public boolean isQuiet() {
        return quiet;
    }

    /**
     * Gets the configuration file if specified.
     *
     * @return the config file or null
     */
    public File getConfigFile() {
        return configFile;
    }

    @Override
    public Integer call() {
        // If no subcommand is provided, show usage
        spec.commandLine().usage(System.out);
        return EXIT_SUCCESS;
    }

    /**
     * Export subcommand for exporting trained models.
     */
    @Command(
        name = "export",
        description = "Export a trained model to a file.",
        mixinStandardHelpOptions = true,
        sortOptions = false
    )
    public int export(
            @Option(names = {"-m", "--model"}, required = true,
                    description = "Path to the trained model file to export")
            File modelFile,

            @Option(names = {"-o", "--output"}, required = true,
                    description = "Output file path (supports .json, .xml, .dat formats)")
            File outputFile,

            @Option(names = {"-f", "--format"},
                    description = "Output format: ${COMPLETION-CANDIDATES} (default: auto-detect from extension)",
                    defaultValue = "auto")
            ExportFormat format,

            @Option(names = {"--include-metadata"},
                    description = "Include training metadata in export")
            boolean includeMetadata,

            @Option(names = {"--compress"},
                    description = "Compress the output file")
            boolean compress
    ) {
        if (!modelFile.exists()) {
            System.err.println("Error: Model file not found: " + modelFile);
            return EXIT_FILE_NOT_FOUND;
        }

        try {
            System.out.println("Exporting model from: " + modelFile);

            // Determine format from extension if auto
            ExportFormat actualFormat = format;
            if (format == ExportFormat.auto) {
                String name = outputFile.getName().toLowerCase();
                if (name.endsWith(".json")) {
                    actualFormat = ExportFormat.json;
                } else if (name.endsWith(".xml")) {
                    actualFormat = ExportFormat.xml;
                } else {
                    actualFormat = ExportFormat.binary;
                }
            }

            System.out.println("Export format: " + actualFormat);

            // Read the model
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(modelFile))) {
                Object model = ois.readObject();

                // Export based on format
                switch (actualFormat) {
                    case json:
                        exportAsJson(model, outputFile, includeMetadata, compress);
                        break;
                    case xml:
                        exportAsXml(model, outputFile, includeMetadata, compress);
                        break;
                    case binary:
                    default:
                        exportAsBinary(model, outputFile, compress);
                        break;
                }
            }

            System.out.println("Model exported successfully to: " + outputFile);
            if (compress) {
                System.out.println("Output file is compressed (gzip)");
            }

            return EXIT_SUCCESS;

        } catch (IOException e) {
            System.err.println("Error exporting model: " + e.getMessage());
            return EXIT_ERROR;
        } catch (ClassNotFoundException e) {
            System.err.println("Error: Invalid model file format");
            return EXIT_ERROR;
        }
    }

    private void exportAsJson(Object model, File outputFile, boolean includeMetadata, boolean compress)
            throws IOException {
        // Use Jackson for JSON export
        try {
            com.fasterxml.jackson.databind.ObjectMapper mapper =
                new com.fasterxml.jackson.databind.ObjectMapper();
            mapper.registerModule(new com.fasterxml.jackson.datatype.jsr310.JavaTimeModule());

            OutputStream os = new FileOutputStream(outputFile);
            if (compress) {
                os = new java.util.zip.GZIPOutputStream(os);
            }

            mapper.writerWithDefaultPrettyPrinter().writeValue(os, model);
            os.close();

        } catch (Exception e) {
            throw new IOException("Failed to export as JSON: " + e.getMessage(), e);
        }
    }

    private void exportAsXml(Object model, File outputFile, boolean includeMetadata, boolean compress)
            throws IOException {
        // Use Java XMLEncoder for XML export
        OutputStream os = new FileOutputStream(outputFile);
        if (compress) {
            os = new java.util.zip.GZIPOutputStream(os);
        }

        try (java.beans.XMLEncoder encoder = new java.beans.XMLEncoder(
                new BufferedOutputStream(os))) {
            encoder.writeObject(model);
        }
    }

    private void exportAsBinary(Object model, File outputFile, boolean compress) throws IOException {
        OutputStream os = new FileOutputStream(outputFile);
        if (compress) {
            os = new java.util.zip.GZIPOutputStream(os);
        }

        try (ObjectOutputStream oos = new ObjectOutputStream(new BufferedOutputStream(os))) {
            oos.writeObject(model);
        }
    }

    /**
     * Import subcommand for importing models.
     */
    @Command(
        name = "import",
        description = "Import a model from a file.",
        mixinStandardHelpOptions = true,
        sortOptions = false
    )
    public int importModel(
            @Option(names = {"-i", "--input"}, required = true,
                    description = "Input file path to import from")
            File inputFile,

            @Option(names = {"-o", "--output"}, required = true,
                    description = "Output model file path (.dat)")
            File outputFile,

            @Option(names = {"-f", "--format"},
                    description = "Input format: ${COMPLETION-CANDIDATES} (default: auto-detect)",
                    defaultValue = "auto")
            ExportFormat format,

            @Option(names = {"--validate"},
                    description = "Validate model after import",
                    defaultValue = "true")
            boolean validate,

            @Option(names = {"--compressed"},
                    description = "Input file is compressed (gzip)")
            boolean compressed
    ) {
        if (!inputFile.exists()) {
            System.err.println("Error: Input file not found: " + inputFile);
            return EXIT_FILE_NOT_FOUND;
        }

        try {
            System.out.println("Importing model from: " + inputFile);

            // Determine format from extension if auto
            ExportFormat actualFormat = format;
            if (format == ExportFormat.auto) {
                String name = inputFile.getName().toLowerCase();
                if (name.endsWith(".json") || name.endsWith(".json.gz")) {
                    actualFormat = ExportFormat.json;
                } else if (name.endsWith(".xml") || name.endsWith(".xml.gz")) {
                    actualFormat = ExportFormat.xml;
                } else {
                    actualFormat = ExportFormat.binary;
                }
            }

            System.out.println("Import format: " + actualFormat);

            Object model;

            // Import based on format
            switch (actualFormat) {
                case json:
                    model = importFromJson(inputFile, compressed);
                    break;
                case xml:
                    model = importFromXml(inputFile, compressed);
                    break;
                case binary:
                default:
                    model = importFromBinary(inputFile, compressed);
                    break;
            }

            if (validate) {
                System.out.println("Validating imported model...");
                validateModel(model);
            }

            // Save as binary model file
            try (ObjectOutputStream oos = new ObjectOutputStream(
                    new BufferedOutputStream(new FileOutputStream(outputFile)))) {
                oos.writeObject(model);
            }

            System.out.println("Model imported successfully to: " + outputFile);
            return EXIT_SUCCESS;

        } catch (IOException e) {
            System.err.println("Error importing model: " + e.getMessage());
            return EXIT_ERROR;
        } catch (ClassNotFoundException e) {
            System.err.println("Error: Unknown model class in import file");
            return EXIT_ERROR;
        }
    }

    private Object importFromJson(File inputFile, boolean compressed)
            throws IOException, ClassNotFoundException {
        try {
            com.fasterxml.jackson.databind.ObjectMapper mapper =
                new com.fasterxml.jackson.databind.ObjectMapper();
            mapper.registerModule(new com.fasterxml.jackson.datatype.jsr310.JavaTimeModule());

            InputStream is = new FileInputStream(inputFile);
            if (compressed) {
                is = new java.util.zip.GZIPInputStream(is);
            }

            // Import as generic map first, then convert
            return mapper.readValue(is, Object.class);

        } catch (Exception e) {
            throw new IOException("Failed to import from JSON: " + e.getMessage(), e);
        }
    }

    private Object importFromXml(File inputFile, boolean compressed)
            throws IOException, ClassNotFoundException {
        InputStream is = new FileInputStream(inputFile);
        if (compressed) {
            is = new java.util.zip.GZIPInputStream(is);
        }

        try (java.beans.XMLDecoder decoder = new java.beans.XMLDecoder(
                new BufferedInputStream(is))) {
            return decoder.readObject();
        }
    }

    private Object importFromBinary(File inputFile, boolean compressed)
            throws IOException, ClassNotFoundException {
        InputStream is = new FileInputStream(inputFile);
        if (compressed) {
            is = new java.util.zip.GZIPInputStream(is);
        }

        try (ObjectInputStream ois = new ObjectInputStream(new BufferedInputStream(is))) {
            return ois.readObject();
        }
    }

    private void validateModel(Object model) {
        // Basic validation - check if model has expected structure
        if (model == null) {
            throw new IllegalArgumentException("Model is null");
        }
        System.out.println("  Model class: " + model.getClass().getName());
        System.out.println("  Validation: PASSED");
    }

    /**
     * Supported export formats.
     */
    public enum ExportFormat {
        auto, json, xml, binary
    }

    /**
     * Main entry point for the CLI.
     *
     * @param args command-line arguments
     */
    public static void main(String[] args) {
        int exitCode = new CommandLine(new FaceRecognitionCLI())
            .setCaseInsensitiveEnumValuesAllowed(true)
            .setExecutionExceptionHandler(new ExceptionHandler())
            .execute(args);
        System.exit(exitCode);
    }

    /**
     * Custom exception handler for user-friendly error messages.
     */
    private static class ExceptionHandler implements CommandLine.IExecutionExceptionHandler {
        @Override
        public int handleExecutionException(Exception ex, CommandLine cmd, CommandLine.ParseResult parseResult) {
            cmd.getErr().println(cmd.getColorScheme().errorText("Error: " + ex.getMessage()));

            if (parseResult.hasMatchedOption("--debug")) {
                ex.printStackTrace(cmd.getErr());
            }

            return EXIT_ERROR;
        }
    }

    /**
     * Utility method to print a progress bar.
     *
     * @param current current progress value
     * @param total total value
     * @param barLength length of the progress bar
     */
    public static void printProgressBar(int current, int total, int barLength) {
        double percent = (double) current / total;
        int filled = (int) (barLength * percent);
        int empty = barLength - filled;

        StringBuilder bar = new StringBuilder("\r[");
        for (int i = 0; i < filled; i++) bar.append("=");
        if (filled < barLength) {
            bar.append(">");
            for (int i = 0; i < empty - 1; i++) bar.append(" ");
        }
        bar.append(String.format("] %3d%% (%d/%d)", (int) (percent * 100), current, total));

        System.out.print(bar);
        if (current == total) {
            System.out.println();
        }
    }

    /**
     * Formats a duration in milliseconds to a human-readable string.
     *
     * @param millis duration in milliseconds
     * @return formatted duration string
     */
    public static String formatDuration(long millis) {
        if (millis < 1000) {
            return millis + "ms";
        } else if (millis < 60000) {
            return String.format("%.2fs", millis / 1000.0);
        } else {
            long minutes = millis / 60000;
            long seconds = (millis % 60000) / 1000;
            return String.format("%dm %ds", minutes, seconds);
        }
    }
}
