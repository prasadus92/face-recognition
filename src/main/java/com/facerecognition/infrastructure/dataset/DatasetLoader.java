package com.facerecognition.infrastructure.dataset;

import com.facerecognition.domain.model.FaceImage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Utility for loading face recognition datasets in various formats.
 *
 * <p>Supports loading datasets in the following formats:</p>
 * <ul>
 *   <li><b>Directory Structure</b>: person_name/image.jpg</li>
 *   <li><b>ORL/AT&amp;T Format</b>: sN/M.pgm</li>
 *   <li><b>Yale Format</b>: subjectNN.condition.gif</li>
 *   <li><b>LFW Pairs Format</b>: pairs.txt with image paths</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * DatasetLoader loader = new DatasetLoader();
 *
 * // Load from directory structure
 * Dataset dataset = loader.loadFromDirectory(Paths.get("faces"));
 *
 * // Load ORL dataset
 * Dataset orlDataset = loader.loadORL(Paths.get("orl_faces"));
 *
 * // Split into train/test
 * TrainTestSplit split = dataset.split(0.8);
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
public class DatasetLoader {

    private static final Logger logger = LoggerFactory.getLogger(DatasetLoader.class);

    private static final Set<String> SUPPORTED_EXTENSIONS = Set.of(
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".pgm", ".ppm"
    );

    /**
     * Represents a loaded dataset with images and labels.
     */
    public static class Dataset {
        private final List<FaceImage> images;
        private final List<String> labels;
        private final Map<String, List<Integer>> labelToIndices;
        private final String name;
        private final Path sourcePath;

        public Dataset(String name, Path sourcePath, List<FaceImage> images, List<String> labels) {
            this.name = name;
            this.sourcePath = sourcePath;
            this.images = Collections.unmodifiableList(new ArrayList<>(images));
            this.labels = Collections.unmodifiableList(new ArrayList<>(labels));

            // Build label index
            this.labelToIndices = new HashMap<>();
            for (int i = 0; i < labels.size(); i++) {
                labelToIndices.computeIfAbsent(labels.get(i), k -> new ArrayList<>()).add(i);
            }
        }

        public List<FaceImage> getImages() { return images; }
        public List<String> getLabels() { return labels; }
        public String getName() { return name; }
        public Path getSourcePath() { return sourcePath; }
        public int size() { return images.size(); }
        public int getNumClasses() { return labelToIndices.size(); }
        public Set<String> getUniqueLabels() { return labelToIndices.keySet(); }

        public List<Integer> getIndicesForLabel(String label) {
            return labelToIndices.getOrDefault(label, Collections.emptyList());
        }

        public FaceImage getImage(int index) { return images.get(index); }
        public String getLabel(int index) { return labels.get(index); }

        /**
         * Splits the dataset into training and testing sets.
         *
         * @param trainRatio the ratio of training samples (0.0 to 1.0)
         * @return the train/test split
         */
        public TrainTestSplit split(double trainRatio) {
            return split(trainRatio, new Random());
        }

        /**
         * Splits the dataset with a specific random seed.
         *
         * @param trainRatio the ratio of training samples
         * @param random the random generator
         * @return the train/test split
         */
        public TrainTestSplit split(double trainRatio, Random random) {
            List<FaceImage> trainImages = new ArrayList<>();
            List<String> trainLabels = new ArrayList<>();
            List<FaceImage> testImages = new ArrayList<>();
            List<String> testLabels = new ArrayList<>();

            // Stratified split - maintain class proportions
            for (String label : labelToIndices.keySet()) {
                List<Integer> indices = new ArrayList<>(labelToIndices.get(label));
                Collections.shuffle(indices, random);

                int trainCount = (int) Math.round(indices.size() * trainRatio);
                for (int i = 0; i < indices.size(); i++) {
                    int idx = indices.get(i);
                    if (i < trainCount) {
                        trainImages.add(images.get(idx));
                        trainLabels.add(labels.get(idx));
                    } else {
                        testImages.add(images.get(idx));
                        testLabels.add(labels.get(idx));
                    }
                }
            }

            Dataset trainSet = new Dataset(name + "_train", sourcePath, trainImages, trainLabels);
            Dataset testSet = new Dataset(name + "_test", sourcePath, testImages, testLabels);

            return new TrainTestSplit(trainSet, testSet);
        }

        /**
         * Creates k folds for cross-validation.
         *
         * @param k the number of folds
         * @return list of train/test splits
         */
        public List<TrainTestSplit> kFold(int k) {
            return kFold(k, new Random());
        }

        /**
         * Creates k folds for cross-validation with a specific seed.
         *
         * @param k the number of folds
         * @param random the random generator
         * @return list of train/test splits
         */
        public List<TrainTestSplit> kFold(int k, Random random) {
            List<TrainTestSplit> folds = new ArrayList<>();

            // Create shuffled indices per class
            Map<String, List<Integer>> shuffledIndices = new HashMap<>();
            for (String label : labelToIndices.keySet()) {
                List<Integer> indices = new ArrayList<>(labelToIndices.get(label));
                Collections.shuffle(indices, random);
                shuffledIndices.put(label, indices);
            }

            for (int fold = 0; fold < k; fold++) {
                List<FaceImage> trainImages = new ArrayList<>();
                List<String> trainLabels = new ArrayList<>();
                List<FaceImage> testImages = new ArrayList<>();
                List<String> testLabels = new ArrayList<>();

                for (String label : shuffledIndices.keySet()) {
                    List<Integer> indices = shuffledIndices.get(label);
                    int foldSize = indices.size() / k;
                    int foldStart = fold * foldSize;
                    int foldEnd = (fold == k - 1) ? indices.size() : (fold + 1) * foldSize;

                    for (int i = 0; i < indices.size(); i++) {
                        int idx = indices.get(i);
                        if (i >= foldStart && i < foldEnd) {
                            testImages.add(images.get(idx));
                            testLabels.add(labels.get(idx));
                        } else {
                            trainImages.add(images.get(idx));
                            trainLabels.add(labels.get(idx));
                        }
                    }
                }

                Dataset trainSet = new Dataset(name + "_fold" + fold + "_train", sourcePath, trainImages, trainLabels);
                Dataset testSet = new Dataset(name + "_fold" + fold + "_test", sourcePath, testImages, testLabels);
                folds.add(new TrainTestSplit(trainSet, testSet));
            }

            return folds;
        }

        @Override
        public String toString() {
            return String.format("Dataset{name='%s', samples=%d, classes=%d}",
                name, images.size(), getNumClasses());
        }
    }

    /**
     * Represents a train/test split of a dataset.
     */
    public static class TrainTestSplit {
        private final Dataset trainSet;
        private final Dataset testSet;

        public TrainTestSplit(Dataset trainSet, Dataset testSet) {
            this.trainSet = trainSet;
            this.testSet = testSet;
        }

        public Dataset getTrainSet() { return trainSet; }
        public Dataset getTestSet() { return testSet; }

        @Override
        public String toString() {
            return String.format("TrainTestSplit{train=%d, test=%d}",
                trainSet.size(), testSet.size());
        }
    }

    /**
     * Loads a dataset from a directory structure.
     * Expected format: basePath/person_name/image.jpg
     *
     * @param basePath the base directory path
     * @return the loaded dataset
     * @throws IOException if loading fails
     */
    public Dataset loadFromDirectory(Path basePath) throws IOException {
        return loadFromDirectory(basePath, basePath.getFileName().toString());
    }

    /**
     * Loads a dataset from a directory structure with a custom name.
     *
     * @param basePath the base directory path
     * @param name the dataset name
     * @return the loaded dataset
     * @throws IOException if loading fails
     */
    public Dataset loadFromDirectory(Path basePath, String name) throws IOException {
        Objects.requireNonNull(basePath, "Base path cannot be null");

        if (!Files.isDirectory(basePath)) {
            throw new IOException("Path is not a directory: " + basePath);
        }

        List<FaceImage> images = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        try (Stream<Path> personDirs = Files.list(basePath)) {
            List<Path> dirs = personDirs
                .filter(Files::isDirectory)
                .sorted()
                .collect(Collectors.toList());

            for (Path personDir : dirs) {
                String personName = personDir.getFileName().toString();

                try (Stream<Path> imageFiles = Files.list(personDir)) {
                    List<Path> files = imageFiles
                        .filter(this::isSupportedImage)
                        .sorted()
                        .collect(Collectors.toList());

                    for (Path imageFile : files) {
                        try {
                            FaceImage image = FaceImage.fromFile(imageFile.toFile());
                            images.add(image);
                            labels.add(personName);
                            logger.debug("Loaded: {} -> {}", imageFile, personName);
                        } catch (Exception e) {
                            logger.warn("Failed to load image: {}", imageFile, e);
                        }
                    }
                }
            }
        }

        logger.info("Loaded dataset '{}' with {} images and {} classes",
            name, images.size(), new HashSet<>(labels).size());

        return new Dataset(name, basePath, images, labels);
    }

    /**
     * Loads the ORL (AT&amp;T) face database.
     * Expected format: basePath/sN/M.pgm
     *
     * @param basePath the base directory path
     * @return the loaded dataset
     * @throws IOException if loading fails
     */
    public Dataset loadORL(Path basePath) throws IOException {
        Objects.requireNonNull(basePath, "Base path cannot be null");

        List<FaceImage> images = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        // ORL has s1-s40 directories
        for (int subject = 1; subject <= 40; subject++) {
            Path subjectDir = basePath.resolve("s" + subject);
            if (!Files.isDirectory(subjectDir)) {
                continue;
            }

            String label = "subject_" + subject;

            // Each subject has 1.pgm to 10.pgm
            for (int imageNum = 1; imageNum <= 10; imageNum++) {
                Path imagePath = subjectDir.resolve(imageNum + ".pgm");
                if (Files.exists(imagePath)) {
                    try {
                        FaceImage image = FaceImage.fromFile(imagePath.toFile());
                        images.add(image);
                        labels.add(label);
                    } catch (Exception e) {
                        logger.warn("Failed to load ORL image: {}", imagePath, e);
                    }
                }
            }
        }

        logger.info("Loaded ORL dataset with {} images and {} subjects",
            images.size(), new HashSet<>(labels).size());

        return new Dataset("ORL", basePath, images, labels);
    }

    /**
     * Loads the Yale face database.
     * Expected format: basePath/subjectNN.condition.gif
     *
     * @param basePath the base directory path
     * @return the loaded dataset
     * @throws IOException if loading fails
     */
    public Dataset loadYale(Path basePath) throws IOException {
        Objects.requireNonNull(basePath, "Base path cannot be null");

        List<FaceImage> images = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        try (Stream<Path> files = Files.list(basePath)) {
            List<Path> imageFiles = files
                .filter(this::isSupportedImage)
                .sorted()
                .collect(Collectors.toList());

            for (Path imagePath : imageFiles) {
                String filename = imagePath.getFileName().toString();

                // Parse Yale format: subjectNN.condition.extension
                String nameWithoutExt = filename.substring(0, filename.lastIndexOf('.'));
                String[] parts = nameWithoutExt.split("\\.");
                if (parts.length >= 1) {
                    // Extract subject identifier
                    String label = parts[0]; // e.g., "subject01"

                    try {
                        FaceImage image = FaceImage.fromFile(imagePath.toFile());
                        images.add(image);
                        labels.add(label);
                    } catch (Exception e) {
                        logger.warn("Failed to load Yale image: {}", imagePath, e);
                    }
                }
            }
        }

        logger.info("Loaded Yale dataset with {} images and {} subjects",
            images.size(), new HashSet<>(labels).size());

        return new Dataset("Yale", basePath, images, labels);
    }

    /**
     * Loads an LFW (Labeled Faces in the Wild) pairs file.
     *
     * @param basePath the base directory containing images
     * @param pairsFile the pairs.txt file
     * @return list of image pairs with match labels
     * @throws IOException if loading fails
     */
    public List<ImagePair> loadLFWPairs(Path basePath, Path pairsFile) throws IOException {
        Objects.requireNonNull(basePath, "Base path cannot be null");
        Objects.requireNonNull(pairsFile, "Pairs file cannot be null");

        List<ImagePair> pairs = new ArrayList<>();

        try (BufferedReader reader = Files.newBufferedReader(pairsFile)) {
            String line;
            int lineNum = 0;

            while ((line = reader.readLine()) != null) {
                lineNum++;
                line = line.trim();
                if (line.isEmpty()) continue;

                String[] parts = line.split("\\s+");

                try {
                    if (parts.length == 3) {
                        // Matched pair: name num1 num2
                        String name = parts[0];
                        int num1 = Integer.parseInt(parts[1]);
                        int num2 = Integer.parseInt(parts[2]);

                        Path img1 = resolveLFWImage(basePath, name, num1);
                        Path img2 = resolveLFWImage(basePath, name, num2);

                        if (Files.exists(img1) && Files.exists(img2)) {
                            FaceImage face1 = FaceImage.fromFile(img1.toFile());
                            FaceImage face2 = FaceImage.fromFile(img2.toFile());
                            pairs.add(new ImagePair(face1, face2, true, name, name));
                        }
                    } else if (parts.length == 4) {
                        // Mismatched pair: name1 num1 name2 num2
                        String name1 = parts[0];
                        int num1 = Integer.parseInt(parts[1]);
                        String name2 = parts[2];
                        int num2 = Integer.parseInt(parts[3]);

                        Path img1 = resolveLFWImage(basePath, name1, num1);
                        Path img2 = resolveLFWImage(basePath, name2, num2);

                        if (Files.exists(img1) && Files.exists(img2)) {
                            FaceImage face1 = FaceImage.fromFile(img1.toFile());
                            FaceImage face2 = FaceImage.fromFile(img2.toFile());
                            pairs.add(new ImagePair(face1, face2, false, name1, name2));
                        }
                    }
                } catch (Exception e) {
                    logger.warn("Failed to parse LFW pair at line {}: {}", lineNum, line, e);
                }
            }
        }

        logger.info("Loaded {} LFW pairs", pairs.size());
        return pairs;
    }

    private Path resolveLFWImage(Path basePath, String name, int num) {
        String filename = String.format("%s_%04d.jpg", name, num);
        return basePath.resolve(name).resolve(filename);
    }

    /**
     * Represents a pair of images for verification testing.
     */
    public static class ImagePair {
        private final FaceImage image1;
        private final FaceImage image2;
        private final boolean matched;
        private final String label1;
        private final String label2;

        public ImagePair(FaceImage image1, FaceImage image2, boolean matched,
                        String label1, String label2) {
            this.image1 = image1;
            this.image2 = image2;
            this.matched = matched;
            this.label1 = label1;
            this.label2 = label2;
        }

        public FaceImage getImage1() { return image1; }
        public FaceImage getImage2() { return image2; }
        public boolean isMatched() { return matched; }
        public String getLabel1() { return label1; }
        public String getLabel2() { return label2; }
    }

    private boolean isSupportedImage(Path path) {
        if (!Files.isRegularFile(path)) return false;
        String name = path.getFileName().toString().toLowerCase();
        return SUPPORTED_EXTENSIONS.stream().anyMatch(name::endsWith);
    }
}
