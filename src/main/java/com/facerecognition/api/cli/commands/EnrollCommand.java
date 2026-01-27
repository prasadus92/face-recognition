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
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * CLI command for enrolling new faces into the recognition system.
 *
 * <p>This command supports enrolling single images or batch enrolling
 * multiple images from directories. Images can be organized by identity
 * in subdirectories or tagged via filename patterns.</p>
 *
 * <h3>Directory Structure Modes:</h3>
 * <ul>
 *   <li><b>--identity</b>: All images belong to specified identity</li>
 *   <li><b>--from-dirs</b>: Subdirectory names become identity names</li>
 *   <li><b>--from-filenames</b>: Extract identity from filename pattern</li>
 * </ul>
 *
 * <h3>Examples:</h3>
 * <pre>
 * # Enroll single image with identity
 * face-recognition enroll -i john.jpg --identity "John Doe" --model model.dat
 *
 * # Enroll all images in directory as same identity
 * face-recognition enroll -i john_photos/ --identity "John Doe" --model model.dat
 *
 * # Enroll from directory structure (subdirectory = identity)
 * face-recognition enroll -i faces/ --from-dirs --model model.dat
 *
 * # Enroll from filename pattern (e.g., "john_001.jpg", "jane_002.jpg")
 * face-recognition enroll -i photos/ --from-filenames --pattern "(.+)_\d+\.jpg"
 * </pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Command(
    name = "enroll",
    aliases = {"add", "register"},
    description = "Enroll new faces into the recognition system.",
    mixinStandardHelpOptions = true,
    sortOptions = false,
    footer = {
        "",
        "Identity Naming Modes:",
        "  --identity NAME    Use specified name for all images",
        "  --from-dirs        Use subdirectory names as identities",
        "  --from-filenames   Extract identity from filename pattern",
        "",
        "Examples:",
        "  @|bold face-recognition enroll -i john.jpg --identity \"John Doe\"|@",
        "  @|bold face-recognition enroll -i faces/ --from-dirs --model model.dat|@",
        "  @|bold face-recognition enroll -i photos/ --from-filenames --pattern \"(.+)_\\d+\"|@"
    }
)
public class EnrollCommand implements Callable<Integer> {

    @ParentCommand
    private FaceRecognitionCLI parent;

    @Parameters(index = "0", arity = "0..1",
            description = "Image file or directory to enroll (alternative to -i)")
    private File inputParam;

    @Option(names = {"-i", "--input"}, paramLabel = "PATH",
            description = "Image file or directory to enroll")
    private File inputFile;

    @Option(names = {"-m", "--model"}, paramLabel = "FILE",
            description = "Path to the model file (creates new if doesn't exist)")
    private File modelFile;

    @Option(names = {"--identity", "--name"}, paramLabel = "NAME",
            description = "Identity name for the enrolled face(s)")
    private String identityName;

    @Option(names = {"--external-id"}, paramLabel = "ID",
            description = "External system ID for the identity")
    private String externalId;

    @Option(names = {"--from-dirs"},
            description = "Use subdirectory names as identity names")
    private boolean fromDirs;

    @Option(names = {"--from-filenames"},
            description = "Extract identity from filename using pattern")
    private boolean fromFilenames;

    @Option(names = {"--pattern"}, paramLabel = "REGEX",
            description = "Regex pattern for extracting identity from filename. " +
                    "First capture group is used. (default: ${DEFAULT-VALUE})",
            defaultValue = "(.+?)(?:_\\d+)?\\.[^.]+$")
    private String filenamePattern;

    @Option(names = {"-r", "--recursive"},
            description = "Process directories recursively")
    private boolean recursive;

    @Option(names = {"-v", "--verbose"},
            description = "Show detailed enrollment information")
    private boolean verbose;

    @Option(names = {"--dry-run"},
            description = "Show what would be enrolled without making changes")
    private boolean dryRun;

    @Option(names = {"--algorithm"}, paramLabel = "ALG",
            description = "Algorithm for new models: ${COMPLETION-CANDIDATES} (default: ${DEFAULT-VALUE})",
            defaultValue = "eigenfaces")
    private Algorithm algorithm;

    @Option(names = {"--components"}, paramLabel = "N",
            description = "Number of components for PCA-based algorithms (default: ${DEFAULT-VALUE})",
            defaultValue = "10")
    private int components;

    @Option(names = {"--extensions"}, paramLabel = "EXT",
            description = "File extensions to process (default: jpg,jpeg,png,bmp)",
            split = ",",
            defaultValue = "jpg,jpeg,png,bmp")
    private Set<String> extensions;

    @Option(names = {"--min-quality"}, paramLabel = "VALUE",
            description = "Minimum image quality score (0.0-1.0, default: ${DEFAULT-VALUE})",
            defaultValue = "0.3")
    private double minQuality;

    @Option(names = {"--skip-duplicates"},
            description = "Skip images that appear to be duplicates")
    private boolean skipDuplicates;

    @Option(names = {"--train-after"},
            description = "Train the model after enrollment (default: true)",
            defaultValue = "true", negatable = true)
    private boolean trainAfter;

    @Option(names = {"--save"},
            description = "Save the model after enrollment (default: true)",
            defaultValue = "true", negatable = true)
    private boolean save;

    /**
     * Supported recognition algorithms.
     */
    public enum Algorithm {
        eigenfaces, lbph, fisherfaces
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

        // Validate options
        int modeCount = 0;
        if (identityName != null) modeCount++;
        if (fromDirs) modeCount++;
        if (fromFilenames) modeCount++;

        if (modeCount == 0 && input.isFile()) {
            System.err.println("Error: Must specify --identity when enrolling a single file");
            return FaceRecognitionCLI.EXIT_INVALID_ARGS;
        }

        if (modeCount > 1) {
            System.err.println("Error: Cannot combine --identity, --from-dirs, and --from-filenames");
            return FaceRecognitionCLI.EXIT_INVALID_ARGS;
        }

        // Compile pattern if needed
        Pattern pattern = null;
        if (fromFilenames) {
            try {
                pattern = Pattern.compile(filenamePattern);
            } catch (Exception e) {
                System.err.println("Error: Invalid regex pattern: " + filenamePattern);
                return FaceRecognitionCLI.EXIT_INVALID_ARGS;
            }
        }

        // Collect files to enroll
        Map<String, List<File>> identityFiles = collectFiles(input, pattern);
        if (identityFiles.isEmpty()) {
            System.err.println("Error: No image files found to enroll");
            return FaceRecognitionCLI.EXIT_FILE_NOT_FOUND;
        }

        int totalFiles = identityFiles.values().stream().mapToInt(List::size).sum();
        int totalIdentities = identityFiles.size();

        if (!isQuiet()) {
            System.out.println("Found " + totalFiles + " image(s) for " + totalIdentities + " identity/identities");
            if (verbose) {
                identityFiles.forEach((id, files) ->
                    System.out.println("  " + id + ": " + files.size() + " image(s)"));
            }
            System.out.println();
        }

        if (dryRun) {
            System.out.println("[DRY RUN] Would enroll:");
            identityFiles.forEach((id, files) -> {
                System.out.println("  Identity: " + id);
                files.forEach(f -> System.out.println("    - " + f.getName()));
            });
            return FaceRecognitionCLI.EXIT_SUCCESS;
        }

        // Load or create model
        FaceRecognitionService service = loadOrCreateService();
        if (service == null) {
            return FaceRecognitionCLI.EXIT_ERROR;
        }

        // Enroll faces
        int enrolled = 0;
        int skipped = 0;
        int errors = 0;
        int current = 0;

        for (Map.Entry<String, List<File>> entry : identityFiles.entrySet()) {
            String identity = entry.getKey();
            List<File> files = entry.getValue();

            for (File file : files) {
                current++;
                if (!isQuiet()) {
                    FaceRecognitionCLI.printProgressBar(current, totalFiles, 40);
                }

                try {
                    FaceImage image = FaceImage.fromFile(file);

                    // Quality check
                    double quality = image.getQualityScore();
                    if (quality < minQuality) {
                        if (verbose) {
                            System.out.printf("\nSkipping %s: quality %.2f < %.2f%n",
                                file.getName(), quality, minQuality);
                        }
                        skipped++;
                        continue;
                    }

                    // Enroll
                    Identity enrolledIdentity = service.enroll(image, identity, externalId);
                    enrolled++;

                    if (verbose) {
                        System.out.printf("\nEnrolled: %s -> %s (quality: %.2f)%n",
                            file.getName(), enrolledIdentity.getName(), quality);
                    }

                } catch (IOException e) {
                    errors++;
                    if (verbose) {
                        System.err.printf("\nError reading %s: %s%n", file.getName(), e.getMessage());
                    }
                } catch (Exception e) {
                    errors++;
                    if (verbose) {
                        System.err.printf("\nError enrolling %s: %s%n", file.getName(), e.getMessage());
                    }
                }
            }
        }

        System.out.println(); // New line after progress bar

        // Train model if requested
        if (trainAfter && enrolled > 0) {
            try {
                if (!isQuiet()) {
                    System.out.println("Training model...");
                }
                long startTime = System.currentTimeMillis();
                service.train();
                long trainTime = System.currentTimeMillis() - startTime;

                if (!isQuiet()) {
                    System.out.println("Training completed in " + FaceRecognitionCLI.formatDuration(trainTime));
                }
            } catch (Exception e) {
                System.err.println("Error training model: " + e.getMessage());
                return FaceRecognitionCLI.EXIT_ERROR;
            }
        }

        // Save model if requested
        if (save && modelFile != null && enrolled > 0) {
            try {
                if (!isQuiet()) {
                    System.out.println("Saving model to: " + modelFile);
                }
                try (ObjectOutputStream oos = new ObjectOutputStream(
                        new BufferedOutputStream(new FileOutputStream(modelFile)))) {
                    oos.writeObject(service);
                }
                if (!isQuiet()) {
                    System.out.println("Model saved successfully");
                }
            } catch (IOException e) {
                System.err.println("Error saving model: " + e.getMessage());
                return FaceRecognitionCLI.EXIT_ERROR;
            }
        }

        // Summary
        if (!isQuiet()) {
            System.out.println("\n" + "=".repeat(50));
            System.out.println("ENROLLMENT SUMMARY");
            System.out.println("=".repeat(50));
            System.out.println("Total images processed: " + totalFiles);
            System.out.println("Successfully enrolled:  " + enrolled);
            System.out.println("Skipped (low quality):  " + skipped);
            System.out.println("Errors:                 " + errors);
            System.out.println("Total identities:       " + service.getIdentityCount());
        }

        return errors == totalFiles ? FaceRecognitionCLI.EXIT_ERROR : FaceRecognitionCLI.EXIT_SUCCESS;
    }

    private boolean isQuiet() {
        return parent != null && parent.isQuiet();
    }

    private Map<String, List<File>> collectFiles(File input, Pattern pattern) {
        Map<String, List<File>> result = new LinkedHashMap<>();

        if (input.isFile()) {
            if (isImageFile(input) && identityName != null) {
                result.computeIfAbsent(identityName, k -> new ArrayList<>()).add(input);
            }
            return result;
        }

        // Directory processing
        try {
            if (fromDirs) {
                // Each subdirectory is an identity
                File[] subdirs = input.listFiles(File::isDirectory);
                if (subdirs != null) {
                    for (File subdir : subdirs) {
                        String identity = subdir.getName();
                        int maxDepth = recursive ? Integer.MAX_VALUE : 1;

                        try (Stream<Path> stream = Files.walk(subdir.toPath(), maxDepth)) {
                            List<File> files = stream
                                .map(Path::toFile)
                                .filter(File::isFile)
                                .filter(this::isImageFile)
                                .sorted()
                                .collect(Collectors.toList());

                            if (!files.isEmpty()) {
                                result.put(identity, files);
                            }
                        }
                    }
                }
            } else {
                // Process all files
                int maxDepth = recursive ? Integer.MAX_VALUE : 1;

                try (Stream<Path> stream = Files.walk(input.toPath(), maxDepth)) {
                    List<File> files = stream
                        .map(Path::toFile)
                        .filter(File::isFile)
                        .filter(this::isImageFile)
                        .sorted()
                        .collect(Collectors.toList());

                    for (File file : files) {
                        String identity;

                        if (identityName != null) {
                            identity = identityName;
                        } else if (fromFilenames && pattern != null) {
                            identity = extractIdentityFromFilename(file.getName(), pattern);
                            if (identity == null) {
                                if (verbose) {
                                    System.err.println("Warning: Could not extract identity from: " + file.getName());
                                }
                                continue;
                            }
                        } else {
                            // Default: use parent directory name
                            identity = file.getParentFile().getName();
                        }

                        result.computeIfAbsent(identity, k -> new ArrayList<>()).add(file);
                    }
                }
            }
        } catch (IOException e) {
            System.err.println("Error scanning directory: " + e.getMessage());
        }

        return result;
    }

    private String extractIdentityFromFilename(String filename, Pattern pattern) {
        Matcher matcher = pattern.matcher(filename);
        if (matcher.find() && matcher.groupCount() >= 1) {
            return matcher.group(1);
        }
        return null;
    }

    private boolean isImageFile(File file) {
        String name = file.getName().toLowerCase();
        return extensions.stream().anyMatch(ext -> name.endsWith("." + ext.toLowerCase()));
    }

    private FaceRecognitionService loadOrCreateService() {
        FaceRecognitionService service = null;

        // Try to load existing model
        if (modelFile != null && modelFile.exists()) {
            try {
                if (!isQuiet()) {
                    System.out.println("Loading existing model from: " + modelFile);
                }

                try (ObjectInputStream ois = new ObjectInputStream(
                        new BufferedInputStream(new FileInputStream(modelFile)))) {
                    Object obj = ois.readObject();

                    if (obj instanceof FaceRecognitionService) {
                        service = (FaceRecognitionService) obj;
                        if (!isQuiet()) {
                            System.out.println("Loaded model with " + service.getIdentityCount() + " existing identities");
                        }
                    }
                }
            } catch (Exception e) {
                System.err.println("Warning: Could not load existing model: " + e.getMessage());
                System.err.println("Creating new model...");
            }
        }

        // Create new service if needed
        if (service == null) {
            if (!isQuiet()) {
                System.out.println("Creating new model with algorithm: " + algorithm);
            }

            FeatureExtractor extractor = createExtractor();
            FaceClassifier classifier = new KNNClassifier();

            service = FaceRecognitionService.builder()
                .extractor(extractor)
                .classifier(classifier)
                .build();
        }

        return service;
    }

    private FeatureExtractor createExtractor() {
        switch (algorithm) {
            case lbph:
                return new LBPHExtractor();
            case fisherfaces:
                return new FisherfacesExtractor(
                    new FeatureExtractor.ExtractorConfig().setNumComponents(components));
            case eigenfaces:
            default:
                return new EigenfacesExtractor(components);
        }
    }
}
