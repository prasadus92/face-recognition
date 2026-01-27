package com.facerecognition.benchmark;

import com.facerecognition.domain.model.FaceImage;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

/**
 * Utility class for loading standard face recognition datasets.
 *
 * <p>This class supports loading faces from various popular dataset formats:</p>
 * <ul>
 *   <li><b>ORL/AT&amp;T Database</b>: 40 subjects, 10 images each, PGM format</li>
 *   <li><b>Yale Face Database</b>: 15 subjects, various expressions and lighting</li>
 *   <li><b>LFW (Labeled Faces in the Wild)</b>: Pairs format for verification</li>
 *   <li><b>Custom Directory</b>: person_name/image.jpg hierarchical format</li>
 * </ul>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * // Load ORL dataset
 * DatasetLoader loader = new DatasetLoader();
 * LoadedDataset dataset = loader.loadOrl("/path/to/orl_faces");
 *
 * // Load custom dataset
 * LoadedDataset custom = loader.loadCustomDirectory("/path/to/faces",
 *     new String[]{".jpg", ".png"});
 *
 * // Get training and test splits
 * DatasetSplit split = dataset.split(0.8, 42); // 80% train, seed 42
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see LoadedDataset
 * @see LabeledFace
 */
public class DatasetLoader {

    private static final String[] DEFAULT_EXTENSIONS = {".jpg", ".jpeg", ".png", ".pgm", ".bmp", ".gif"};

    private int targetWidth = 48;
    private int targetHeight = 64;
    private boolean normalizeSize = true;
    private boolean convertToGrayscale = false;

    /**
     * Represents a loaded face dataset.
     */
    public static class LoadedDataset implements Serializable {
        private static final long serialVersionUID = 1L;

        private final String name;
        private final String format;
        private final Path sourcePath;
        private final List<LabeledFace> faces;
        private final Map<String, List<LabeledFace>> facesByLabel;
        private final int imageWidth;
        private final int imageHeight;

        public LoadedDataset(String name, String format, Path sourcePath,
                            List<LabeledFace> faces, int width, int height) {
            this.name = name;
            this.format = format;
            this.sourcePath = sourcePath;
            this.faces = new ArrayList<>(faces);
            this.imageWidth = width;
            this.imageHeight = height;

            // Group by label
            this.facesByLabel = faces.stream()
                .collect(Collectors.groupingBy(LabeledFace::getLabel));
        }

        public String getName() { return name; }
        public String getFormat() { return format; }
        public Path getSourcePath() { return sourcePath; }
        public List<LabeledFace> getFaces() { return Collections.unmodifiableList(faces); }
        public int getImageWidth() { return imageWidth; }
        public int getImageHeight() { return imageHeight; }

        /**
         * Gets the number of unique classes (identities).
         *
         * @return number of classes
         */
        public int getNumClasses() {
            return facesByLabel.size();
        }

        /**
         * Gets all unique labels in the dataset.
         *
         * @return set of labels
         */
        public Set<String> getLabels() {
            return facesByLabel.keySet();
        }

        /**
         * Gets faces for a specific label.
         *
         * @param label the label to filter by
         * @return list of faces with that label
         */
        public List<LabeledFace> getFacesByLabel(String label) {
            return facesByLabel.getOrDefault(label, Collections.emptyList());
        }

        /**
         * Gets the number of images per class (assumes uniform distribution).
         *
         * @return average images per class
         */
        public int getImagesPerClass() {
            return faces.size() / Math.max(1, getNumClasses());
        }

        /**
         * Splits the dataset into training and test sets.
         *
         * @param trainRatio ratio of data for training (0.0 to 1.0)
         * @param seed random seed for reproducibility
         * @return the dataset split
         */
        public DatasetSplit split(double trainRatio, long seed) {
            return DatasetSplit.randomSplit(this, trainRatio, seed);
        }

        /**
         * Splits using leave-one-out per class.
         *
         * @param leaveOutIndex index of sample to leave out from each class
         * @return the dataset split
         */
        public DatasetSplit leaveOneOut(int leaveOutIndex) {
            return DatasetSplit.leaveOneOut(this, leaveOutIndex);
        }

        /**
         * Creates K-fold cross validation splits.
         *
         * @param k number of folds
         * @param seed random seed
         * @return list of k dataset splits
         */
        public List<DatasetSplit> kFoldSplit(int k, long seed) {
            return DatasetSplit.kFold(this, k, seed);
        }

        /**
         * Creates stratified K-fold cross validation splits.
         *
         * @param k number of folds
         * @param seed random seed
         * @return list of k dataset splits
         */
        public List<DatasetSplit> stratifiedKFold(int k, long seed) {
            return DatasetSplit.stratifiedKFold(this, k, seed);
        }

        /**
         * Creates BenchmarkResult.DatasetInfo from this dataset.
         *
         * @param trainSize number of training samples
         * @param testSize number of test samples
         * @return dataset info
         */
        public BenchmarkResult.DatasetInfo toDatasetInfo(int trainSize, int testSize) {
            return new BenchmarkResult.DatasetInfo(
                name, format, faces.size(), getNumClasses(),
                getImagesPerClass(), imageWidth, imageHeight,
                trainSize, testSize
            );
        }

        @Override
        public String toString() {
            return String.format("LoadedDataset{name='%s', format='%s', classes=%d, faces=%d, size=%dx%d}",
                name, format, getNumClasses(), faces.size(), imageWidth, imageHeight);
        }
    }

    /**
     * Represents a face image with its associated label.
     */
    public static class LabeledFace implements Serializable {
        private static final long serialVersionUID = 1L;

        private final FaceImage image;
        private final String label;
        private final Path sourcePath;
        private final Map<String, String> metadata;

        public LabeledFace(FaceImage image, String label, Path sourcePath) {
            this.image = image;
            this.label = label;
            this.sourcePath = sourcePath;
            this.metadata = new HashMap<>();
        }

        public FaceImage getImage() { return image; }
        public String getLabel() { return label; }
        public Path getSourcePath() { return sourcePath; }
        public Map<String, String> getMetadata() { return metadata; }

        public void setMetadata(String key, String value) {
            metadata.put(key, value);
        }

        @Override
        public String toString() {
            return String.format("LabeledFace{label='%s', source='%s'}",
                label, sourcePath.getFileName());
        }
    }

    /**
     * Represents a train/test split of a dataset.
     */
    public static class DatasetSplit implements Serializable {
        private static final long serialVersionUID = 1L;

        private final List<LabeledFace> trainSet;
        private final List<LabeledFace> testSet;
        private final int foldIndex;
        private final int totalFolds;

        public DatasetSplit(List<LabeledFace> trainSet, List<LabeledFace> testSet) {
            this(trainSet, testSet, 0, 1);
        }

        public DatasetSplit(List<LabeledFace> trainSet, List<LabeledFace> testSet,
                           int foldIndex, int totalFolds) {
            this.trainSet = new ArrayList<>(trainSet);
            this.testSet = new ArrayList<>(testSet);
            this.foldIndex = foldIndex;
            this.totalFolds = totalFolds;
        }

        public List<LabeledFace> getTrainSet() { return Collections.unmodifiableList(trainSet); }
        public List<LabeledFace> getTestSet() { return Collections.unmodifiableList(testSet); }
        public int getTrainSize() { return trainSet.size(); }
        public int getTestSize() { return testSet.size(); }
        public int getFoldIndex() { return foldIndex; }
        public int getTotalFolds() { return totalFolds; }

        /**
         * Gets training images only.
         *
         * @return list of training FaceImages
         */
        public List<FaceImage> getTrainImages() {
            return trainSet.stream()
                .map(LabeledFace::getImage)
                .collect(Collectors.toList());
        }

        /**
         * Gets training labels only.
         *
         * @return list of training labels
         */
        public List<String> getTrainLabels() {
            return trainSet.stream()
                .map(LabeledFace::getLabel)
                .collect(Collectors.toList());
        }

        /**
         * Gets test images only.
         *
         * @return list of test FaceImages
         */
        public List<FaceImage> getTestImages() {
            return testSet.stream()
                .map(LabeledFace::getImage)
                .collect(Collectors.toList());
        }

        /**
         * Gets test labels only.
         *
         * @return list of test labels
         */
        public List<String> getTestLabels() {
            return testSet.stream()
                .map(LabeledFace::getLabel)
                .collect(Collectors.toList());
        }

        /**
         * Creates a random train/test split.
         */
        public static DatasetSplit randomSplit(LoadedDataset dataset, double trainRatio, long seed) {
            List<LabeledFace> all = new ArrayList<>(dataset.getFaces());
            Random random = new Random(seed);
            Collections.shuffle(all, random);

            int trainSize = (int) (all.size() * trainRatio);
            List<LabeledFace> train = all.subList(0, trainSize);
            List<LabeledFace> test = all.subList(trainSize, all.size());

            return new DatasetSplit(train, test);
        }

        /**
         * Creates a leave-one-out split per class.
         */
        public static DatasetSplit leaveOneOut(LoadedDataset dataset, int leaveOutIndex) {
            List<LabeledFace> train = new ArrayList<>();
            List<LabeledFace> test = new ArrayList<>();

            for (String label : dataset.getLabels()) {
                List<LabeledFace> classFaces = dataset.getFacesByLabel(label);
                for (int i = 0; i < classFaces.size(); i++) {
                    if (i == leaveOutIndex % classFaces.size()) {
                        test.add(classFaces.get(i));
                    } else {
                        train.add(classFaces.get(i));
                    }
                }
            }

            return new DatasetSplit(train, test);
        }

        /**
         * Creates K-fold cross validation splits.
         */
        public static List<DatasetSplit> kFold(LoadedDataset dataset, int k, long seed) {
            List<LabeledFace> all = new ArrayList<>(dataset.getFaces());
            Random random = new Random(seed);
            Collections.shuffle(all, random);

            List<DatasetSplit> folds = new ArrayList<>();
            int foldSize = all.size() / k;

            for (int i = 0; i < k; i++) {
                int testStart = i * foldSize;
                int testEnd = (i == k - 1) ? all.size() : (i + 1) * foldSize;

                List<LabeledFace> test = all.subList(testStart, testEnd);
                List<LabeledFace> train = new ArrayList<>();
                train.addAll(all.subList(0, testStart));
                train.addAll(all.subList(testEnd, all.size()));

                folds.add(new DatasetSplit(train, test, i, k));
            }

            return folds;
        }

        /**
         * Creates stratified K-fold cross validation splits.
         * Maintains class distribution in each fold.
         */
        public static List<DatasetSplit> stratifiedKFold(LoadedDataset dataset, int k, long seed) {
            Random random = new Random(seed);
            List<DatasetSplit> folds = new ArrayList<>();

            // Group and shuffle within each class
            Map<String, List<LabeledFace>> byClass = new HashMap<>();
            for (String label : dataset.getLabels()) {
                List<LabeledFace> classFaces = new ArrayList<>(dataset.getFacesByLabel(label));
                Collections.shuffle(classFaces, random);
                byClass.put(label, classFaces);
            }

            // Create k folds
            for (int fold = 0; fold < k; fold++) {
                List<LabeledFace> train = new ArrayList<>();
                List<LabeledFace> test = new ArrayList<>();

                for (String label : byClass.keySet()) {
                    List<LabeledFace> classFaces = byClass.get(label);
                    int foldSize = classFaces.size() / k;
                    int testStart = fold * foldSize;
                    int testEnd = (fold == k - 1) ? classFaces.size() : (fold + 1) * foldSize;

                    for (int i = 0; i < classFaces.size(); i++) {
                        if (i >= testStart && i < testEnd) {
                            test.add(classFaces.get(i));
                        } else {
                            train.add(classFaces.get(i));
                        }
                    }
                }

                folds.add(new DatasetSplit(train, test, fold, k));
            }

            return folds;
        }

        @Override
        public String toString() {
            return String.format("DatasetSplit{fold=%d/%d, train=%d, test=%d}",
                foldIndex + 1, totalFolds, trainSet.size(), testSet.size());
        }
    }

    /**
     * Represents a verification pair for LFW-style evaluation.
     */
    public static class VerificationPair implements Serializable {
        private static final long serialVersionUID = 1L;

        private final LabeledFace face1;
        private final LabeledFace face2;
        private final boolean samePerson;

        public VerificationPair(LabeledFace face1, LabeledFace face2, boolean samePerson) {
            this.face1 = face1;
            this.face2 = face2;
            this.samePerson = samePerson;
        }

        public LabeledFace getFace1() { return face1; }
        public LabeledFace getFace2() { return face2; }
        public boolean isSamePerson() { return samePerson; }

        @Override
        public String toString() {
            return String.format("VerificationPair{person1='%s', person2='%s', same=%s}",
                face1.getLabel(), face2.getLabel(), samePerson);
        }
    }

    /**
     * Creates a new DatasetLoader with default settings.
     */
    public DatasetLoader() {
    }

    /**
     * Sets the target image size for normalization.
     *
     * @param width target width
     * @param height target height
     * @return this loader for chaining
     */
    public DatasetLoader setTargetSize(int width, int height) {
        this.targetWidth = width;
        this.targetHeight = height;
        return this;
    }

    /**
     * Sets whether to normalize image sizes.
     *
     * @param normalize true to normalize
     * @return this loader for chaining
     */
    public DatasetLoader setNormalizeSize(boolean normalize) {
        this.normalizeSize = normalize;
        return this;
    }

    /**
     * Sets whether to convert images to grayscale.
     *
     * @param grayscale true to convert
     * @return this loader for chaining
     */
    public DatasetLoader setConvertToGrayscale(boolean grayscale) {
        this.convertToGrayscale = grayscale;
        return this;
    }

    /**
     * Loads the ORL/AT&amp;T Face Database.
     *
     * <p>Expected directory structure:</p>
     * <pre>
     * orl_faces/
     *   s1/
     *     1.pgm
     *     2.pgm
     *     ...
     *   s2/
     *     1.pgm
     *     ...
     * </pre>
     *
     * @param path path to the ORL dataset directory
     * @return loaded dataset
     * @throws IOException if loading fails
     */
    public LoadedDataset loadOrl(String path) throws IOException {
        return loadOrl(Paths.get(path));
    }

    /**
     * Loads the ORL/AT&amp;T Face Database.
     *
     * @param path path to the ORL dataset directory
     * @return loaded dataset
     * @throws IOException if loading fails
     */
    public LoadedDataset loadOrl(Path path) throws IOException {
        if (!Files.exists(path)) {
            throw new IOException("ORL dataset path does not exist: " + path);
        }

        List<LabeledFace> faces = new ArrayList<>();
        int width = 92, height = 112;  // Default ORL size

        try (Stream<Path> subjects = Files.list(path)) {
            List<Path> subjectDirs = subjects
                .filter(Files::isDirectory)
                .filter(p -> p.getFileName().toString().startsWith("s"))
                .sorted()
                .collect(Collectors.toList());

            for (Path subjectDir : subjectDirs) {
                String label = subjectDir.getFileName().toString();

                try (Stream<Path> images = Files.list(subjectDir)) {
                    List<Path> imageFiles = images
                        .filter(p -> p.toString().toLowerCase().endsWith(".pgm"))
                        .sorted()
                        .collect(Collectors.toList());

                    for (Path imageFile : imageFiles) {
                        FaceImage face = loadImage(imageFile);
                        if (face != null) {
                            faces.add(new LabeledFace(face, label, imageFile));
                            if (faces.size() == 1) {
                                width = face.getWidth();
                                height = face.getHeight();
                            }
                        }
                    }
                }
            }
        }

        return new LoadedDataset("ORL/AT&T", "orl", path, faces, width, height);
    }

    /**
     * Loads the Yale Face Database.
     *
     * <p>Expected structure with filenames like: subject01.glasses.pgm</p>
     *
     * @param path path to the Yale dataset directory
     * @return loaded dataset
     * @throws IOException if loading fails
     */
    public LoadedDataset loadYale(String path) throws IOException {
        return loadYale(Paths.get(path));
    }

    /**
     * Loads the Yale Face Database.
     *
     * @param path path to the Yale dataset directory
     * @return loaded dataset
     * @throws IOException if loading fails
     */
    public LoadedDataset loadYale(Path path) throws IOException {
        if (!Files.exists(path)) {
            throw new IOException("Yale dataset path does not exist: " + path);
        }

        List<LabeledFace> faces = new ArrayList<>();
        int width = 0, height = 0;

        try (Stream<Path> files = Files.list(path)) {
            List<Path> imageFiles = files
                .filter(p -> isImageFile(p.toString()))
                .sorted()
                .collect(Collectors.toList());

            for (Path imageFile : imageFiles) {
                String filename = imageFile.getFileName().toString().toLowerCase();
                String label = extractYaleLabel(filename);

                if (label != null) {
                    FaceImage face = loadImage(imageFile);
                    if (face != null) {
                        LabeledFace labeled = new LabeledFace(face, label, imageFile);

                        // Extract expression/condition from filename
                        String condition = extractYaleCondition(filename);
                        if (condition != null) {
                            labeled.setMetadata("condition", condition);
                        }

                        faces.add(labeled);
                        if (width == 0) {
                            width = face.getWidth();
                            height = face.getHeight();
                        }
                    }
                }
            }
        }

        return new LoadedDataset("Yale", "yale", path, faces, width, height);
    }

    /**
     * Loads LFW (Labeled Faces in the Wild) pairs format.
     *
     * <p>Expected structure:</p>
     * <pre>
     * lfw/
     *   pairs.txt          # Pair definitions
     *   Aaron_Eckhart/
     *     Aaron_Eckhart_0001.jpg
     *     ...
     *   Adam_Brody/
     *     ...
     * </pre>
     *
     * @param dataPath path to the LFW image directory
     * @param pairsFile path to the pairs.txt file
     * @return list of verification pairs
     * @throws IOException if loading fails
     */
    public List<VerificationPair> loadLfwPairs(String dataPath, String pairsFile) throws IOException {
        return loadLfwPairs(Paths.get(dataPath), Paths.get(pairsFile));
    }

    /**
     * Loads LFW pairs format.
     *
     * @param dataPath path to the LFW image directory
     * @param pairsFile path to the pairs.txt file
     * @return list of verification pairs
     * @throws IOException if loading fails
     */
    public List<VerificationPair> loadLfwPairs(Path dataPath, Path pairsFile) throws IOException {
        if (!Files.exists(dataPath)) {
            throw new IOException("LFW data path does not exist: " + dataPath);
        }
        if (!Files.exists(pairsFile)) {
            throw new IOException("Pairs file does not exist: " + pairsFile);
        }

        List<VerificationPair> pairs = new ArrayList<>();
        List<String> lines = Files.readAllLines(pairsFile);

        // First line contains metadata (number of folds, pairs per fold)
        int lineIndex = 1;

        while (lineIndex < lines.size()) {
            String line = lines.get(lineIndex).trim();
            if (line.isEmpty()) {
                lineIndex++;
                continue;
            }

            String[] parts = line.split("\\s+");

            if (parts.length == 3) {
                // Same person pair: name, num1, num2
                String name = parts[0];
                int num1 = Integer.parseInt(parts[1]);
                int num2 = Integer.parseInt(parts[2]);

                LabeledFace face1 = loadLfwFace(dataPath, name, num1);
                LabeledFace face2 = loadLfwFace(dataPath, name, num2);

                if (face1 != null && face2 != null) {
                    pairs.add(new VerificationPair(face1, face2, true));
                }
            } else if (parts.length == 4) {
                // Different person pair: name1, num1, name2, num2
                String name1 = parts[0];
                int num1 = Integer.parseInt(parts[1]);
                String name2 = parts[2];
                int num2 = Integer.parseInt(parts[3]);

                LabeledFace face1 = loadLfwFace(dataPath, name1, num1);
                LabeledFace face2 = loadLfwFace(dataPath, name2, num2);

                if (face1 != null && face2 != null) {
                    pairs.add(new VerificationPair(face1, face2, false));
                }
            }

            lineIndex++;
        }

        return pairs;
    }

    /**
     * Loads a complete LFW dataset as a standard dataset.
     *
     * @param path path to the LFW directory
     * @return loaded dataset
     * @throws IOException if loading fails
     */
    public LoadedDataset loadLfw(String path) throws IOException {
        return loadLfw(Paths.get(path));
    }

    /**
     * Loads a complete LFW dataset.
     *
     * @param path path to the LFW directory
     * @return loaded dataset
     * @throws IOException if loading fails
     */
    public LoadedDataset loadLfw(Path path) throws IOException {
        if (!Files.exists(path)) {
            throw new IOException("LFW path does not exist: " + path);
        }

        List<LabeledFace> faces = new ArrayList<>();
        int width = 0, height = 0;

        try (Stream<Path> persons = Files.list(path)) {
            List<Path> personDirs = persons
                .filter(Files::isDirectory)
                .sorted()
                .collect(Collectors.toList());

            for (Path personDir : personDirs) {
                String label = personDir.getFileName().toString().replace("_", " ");

                try (Stream<Path> images = Files.list(personDir)) {
                    List<Path> imageFiles = images
                        .filter(p -> isImageFile(p.toString()))
                        .sorted()
                        .collect(Collectors.toList());

                    for (Path imageFile : imageFiles) {
                        FaceImage face = loadImage(imageFile);
                        if (face != null) {
                            faces.add(new LabeledFace(face, label, imageFile));
                            if (width == 0) {
                                width = face.getWidth();
                                height = face.getHeight();
                            }
                        }
                    }
                }
            }
        }

        return new LoadedDataset("LFW", "lfw", path, faces, width, height);
    }

    /**
     * Loads a custom dataset with hierarchical directory structure.
     *
     * <p>Expected structure:</p>
     * <pre>
     * dataset/
     *   person1/
     *     image1.jpg
     *     image2.jpg
     *   person2/
     *     image1.jpg
     *     ...
     * </pre>
     *
     * @param path path to the dataset directory
     * @return loaded dataset
     * @throws IOException if loading fails
     */
    public LoadedDataset loadCustomDirectory(String path) throws IOException {
        return loadCustomDirectory(Paths.get(path), DEFAULT_EXTENSIONS);
    }

    /**
     * Loads a custom dataset with specified file extensions.
     *
     * @param path path to the dataset directory
     * @param extensions allowed file extensions (e.g., ".jpg", ".png")
     * @return loaded dataset
     * @throws IOException if loading fails
     */
    public LoadedDataset loadCustomDirectory(String path, String[] extensions) throws IOException {
        return loadCustomDirectory(Paths.get(path), extensions);
    }

    /**
     * Loads a custom dataset with hierarchical directory structure.
     *
     * @param path path to the dataset directory
     * @param extensions allowed file extensions
     * @return loaded dataset
     * @throws IOException if loading fails
     */
    public LoadedDataset loadCustomDirectory(Path path, String[] extensions) throws IOException {
        if (!Files.exists(path)) {
            throw new IOException("Dataset path does not exist: " + path);
        }

        Set<String> extSet = new HashSet<>();
        for (String ext : extensions) {
            extSet.add(ext.toLowerCase());
        }

        List<LabeledFace> faces = new ArrayList<>();
        int width = 0, height = 0;
        String datasetName = path.getFileName().toString();

        try (Stream<Path> persons = Files.list(path)) {
            List<Path> personDirs = persons
                .filter(Files::isDirectory)
                .sorted()
                .collect(Collectors.toList());

            for (Path personDir : personDirs) {
                String label = personDir.getFileName().toString();

                try (Stream<Path> images = Files.list(personDir)) {
                    List<Path> imageFiles = images
                        .filter(p -> hasExtension(p.toString(), extSet))
                        .sorted()
                        .collect(Collectors.toList());

                    for (Path imageFile : imageFiles) {
                        FaceImage face = loadImage(imageFile);
                        if (face != null) {
                            faces.add(new LabeledFace(face, label, imageFile));
                            if (width == 0) {
                                width = face.getWidth();
                                height = face.getHeight();
                            }
                        }
                    }
                }
            }
        }

        return new LoadedDataset(datasetName, "custom", path, faces, width, height);
    }

    /**
     * Loads a flat directory where label is extracted from filename.
     *
     * <p>Filename format: label_index.ext (e.g., "person1_01.jpg")</p>
     *
     * @param path path to the directory
     * @param labelSeparator character separating label from index
     * @return loaded dataset
     * @throws IOException if loading fails
     */
    public LoadedDataset loadFlatDirectory(String path, String labelSeparator) throws IOException {
        return loadFlatDirectory(Paths.get(path), labelSeparator);
    }

    /**
     * Loads a flat directory dataset.
     *
     * @param path path to the directory
     * @param labelSeparator character separating label from index
     * @return loaded dataset
     * @throws IOException if loading fails
     */
    public LoadedDataset loadFlatDirectory(Path path, String labelSeparator) throws IOException {
        if (!Files.exists(path)) {
            throw new IOException("Dataset path does not exist: " + path);
        }

        List<LabeledFace> faces = new ArrayList<>();
        int width = 0, height = 0;
        String datasetName = path.getFileName().toString();

        try (Stream<Path> files = Files.list(path)) {
            List<Path> imageFiles = files
                .filter(p -> isImageFile(p.toString()))
                .sorted()
                .collect(Collectors.toList());

            for (Path imageFile : imageFiles) {
                String filename = imageFile.getFileName().toString();
                String nameWithoutExt = filename.substring(0, filename.lastIndexOf('.'));
                int sepIndex = nameWithoutExt.lastIndexOf(labelSeparator);

                String label;
                if (sepIndex > 0) {
                    label = nameWithoutExt.substring(0, sepIndex);
                } else {
                    label = nameWithoutExt;
                }

                FaceImage face = loadImage(imageFile);
                if (face != null) {
                    faces.add(new LabeledFace(face, label, imageFile));
                    if (width == 0) {
                        width = face.getWidth();
                        height = face.getHeight();
                    }
                }
            }
        }

        return new LoadedDataset(datasetName, "flat", path, faces, width, height);
    }

    /**
     * Loads an image and optionally normalizes it.
     */
    private FaceImage loadImage(Path path) {
        try {
            BufferedImage image;

            if (path.toString().toLowerCase().endsWith(".pgm")) {
                image = loadPgm(path);
            } else {
                image = ImageIO.read(path.toFile());
            }

            if (image == null) {
                return null;
            }

            // Convert to grayscale if requested
            if (convertToGrayscale && image.getType() != BufferedImage.TYPE_BYTE_GRAY) {
                BufferedImage gray = new BufferedImage(
                    image.getWidth(), image.getHeight(), BufferedImage.TYPE_BYTE_GRAY);
                gray.getGraphics().drawImage(image, 0, 0, null);
                image = gray;
            }

            FaceImage faceImage = FaceImage.fromBufferedImage(image);

            // Resize if normalization is enabled
            if (normalizeSize &&
                (faceImage.getWidth() != targetWidth || faceImage.getHeight() != targetHeight)) {
                faceImage = faceImage.resize(targetWidth, targetHeight);
            }

            return faceImage;
        } catch (Exception e) {
            System.err.println("Warning: Failed to load image: " + path + " - " + e.getMessage());
            return null;
        }
    }

    /**
     * Loads a PGM (Portable Gray Map) image.
     */
    private BufferedImage loadPgm(Path path) throws IOException {
        try (DataInputStream dis = new DataInputStream(
                new BufferedInputStream(Files.newInputStream(path)))) {

            // Read magic number
            String magic = readLine(dis);
            if (!magic.equals("P5")) {
                // Try P2 (ASCII) format
                if (magic.equals("P2")) {
                    return loadPgmAscii(path);
                }
                throw new IOException("Not a valid PGM file (expected P5 or P2): " + magic);
            }

            // Skip comments
            String line;
            do {
                line = readLine(dis);
            } while (line.startsWith("#"));

            // Parse dimensions
            String[] dims = line.split("\\s+");
            int width = Integer.parseInt(dims[0]);
            int height = Integer.parseInt(dims[1]);

            // Read max value
            String maxLine = readLine(dis);
            int maxVal = Integer.parseInt(maxLine.trim());

            // Read pixel data
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
            byte[] pixels = new byte[width * height];

            if (maxVal <= 255) {
                dis.readFully(pixels);
            } else {
                // 16-bit values, scale to 8-bit
                for (int i = 0; i < pixels.length; i++) {
                    int val = dis.readUnsignedShort();
                    pixels[i] = (byte) (val * 255 / maxVal);
                }
            }

            image.getRaster().setDataElements(0, 0, width, height, pixels);
            return image;
        }
    }

    /**
     * Loads a PGM file in ASCII format (P2).
     */
    private BufferedImage loadPgmAscii(Path path) throws IOException {
        List<String> lines = Files.readAllLines(path);
        int lineIndex = 1;

        // Skip comments
        while (lines.get(lineIndex).startsWith("#")) {
            lineIndex++;
        }

        // Parse dimensions
        String[] dims = lines.get(lineIndex++).split("\\s+");
        int width = Integer.parseInt(dims[0]);
        int height = Integer.parseInt(dims[1]);

        // Skip max value line
        lineIndex++;

        // Read pixel values
        BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_BYTE_GRAY);
        StringBuilder allValues = new StringBuilder();
        while (lineIndex < lines.size()) {
            allValues.append(lines.get(lineIndex++)).append(" ");
        }

        String[] values = allValues.toString().trim().split("\\s+");
        byte[] pixels = new byte[width * height];
        for (int i = 0; i < pixels.length && i < values.length; i++) {
            pixels[i] = (byte) Integer.parseInt(values[i]);
        }

        image.getRaster().setDataElements(0, 0, width, height, pixels);
        return image;
    }

    private String readLine(DataInputStream dis) throws IOException {
        StringBuilder sb = new StringBuilder();
        int c;
        while ((c = dis.read()) != -1 && c != '\n') {
            if (c != '\r') {
                sb.append((char) c);
            }
        }
        return sb.toString();
    }

    private LabeledFace loadLfwFace(Path dataPath, String name, int number) {
        String filename = String.format("%s_%04d.jpg", name, number);
        Path imagePath = dataPath.resolve(name).resolve(filename);

        if (!Files.exists(imagePath)) {
            return null;
        }

        FaceImage image = loadImage(imagePath);
        if (image == null) {
            return null;
        }

        return new LabeledFace(image, name.replace("_", " "), imagePath);
    }

    private String extractYaleLabel(String filename) {
        // Yale format: subjectXX.expression.pgm
        if (filename.startsWith("subject")) {
            int dotIndex = filename.indexOf('.');
            if (dotIndex > 0) {
                return filename.substring(0, dotIndex);
            }
        }
        return null;
    }

    private String extractYaleCondition(String filename) {
        // Yale format: subjectXX.expression.pgm
        int firstDot = filename.indexOf('.');
        int lastDot = filename.lastIndexOf('.');
        if (firstDot > 0 && lastDot > firstDot) {
            return filename.substring(firstDot + 1, lastDot);
        }
        return null;
    }

    private boolean isImageFile(String filename) {
        String lower = filename.toLowerCase();
        for (String ext : DEFAULT_EXTENSIONS) {
            if (lower.endsWith(ext)) {
                return true;
            }
        }
        return false;
    }

    private boolean hasExtension(String filename, Set<String> extensions) {
        String lower = filename.toLowerCase();
        for (String ext : extensions) {
            if (lower.endsWith(ext)) {
                return true;
            }
        }
        return false;
    }
}
