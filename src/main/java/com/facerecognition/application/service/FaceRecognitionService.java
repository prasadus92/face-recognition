package com.facerecognition.application.service;

import com.facerecognition.domain.model.*;
import com.facerecognition.domain.service.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Main application service for face recognition operations.
 *
 * <p>This service orchestrates the complete face recognition pipeline:</p>
 * <ol>
 *   <li>Face Detection - Locate faces in images</li>
 *   <li>Face Alignment - Normalize face orientation and size</li>
 *   <li>Feature Extraction - Convert faces to feature vectors</li>
 *   <li>Classification - Match against enrolled identities</li>
 * </ol>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * // Build the service
 * FaceRecognitionService service = FaceRecognitionService.builder()
 *     .detector(new HaarCascadeDetector())
 *     .extractor(new EigenfacesExtractor(10))
 *     .classifier(new KNNClassifier())
 *     .build();
 *
 * // Enroll faces
 * service.enroll(FaceImage.fromFile(new File("john.jpg")), "John Doe");
 * service.enroll(FaceImage.fromFile(new File("jane.jpg")), "Jane Doe");
 *
 * // Train the system
 * service.train();
 *
 * // Recognize a face
 * RecognitionResult result = service.recognize(FaceImage.fromFile(new File("unknown.jpg")));
 * System.out.println("Recognized: " + result.getIdentity().map(Identity::getName).orElse("Unknown"));
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
public class FaceRecognitionService {

    private final FaceDetector detector;
    private final FeatureExtractor extractor;
    private final FaceClassifier classifier;
    private final Map<String, Identity> identities;
    private final List<TrainingSample> trainingSamples;
    private final Config config;

    private boolean trained;

    /**
     * Training sample container.
     */
    private static class TrainingSample {
        final FaceImage image;
        final String identityId;
        final String label;

        TrainingSample(FaceImage image, String identityId, String label) {
            this.image = image;
            this.identityId = identityId;
            this.label = label;
        }
    }

    /**
     * Service configuration.
     */
    public static class Config {
        private double recognitionThreshold = 0.6;
        private double detectionConfidence = 0.5;
        private double minQuality = 0.3;
        private boolean autoAlign = true;
        private int targetWidth = 48;
        private int targetHeight = 64;

        public double getRecognitionThreshold() { return recognitionThreshold; }
        public Config setRecognitionThreshold(double t) { this.recognitionThreshold = t; return this; }

        public double getDetectionConfidence() { return detectionConfidence; }
        public Config setDetectionConfidence(double c) { this.detectionConfidence = c; return this; }

        public double getMinQuality() { return minQuality; }
        public Config setMinQuality(double q) { this.minQuality = q; return this; }

        public boolean isAutoAlign() { return autoAlign; }
        public Config setAutoAlign(boolean a) { this.autoAlign = a; return this; }

        public int getTargetWidth() { return targetWidth; }
        public Config setTargetWidth(int w) { this.targetWidth = w; return this; }

        public int getTargetHeight() { return targetHeight; }
        public Config setTargetHeight(int h) { this.targetHeight = h; return this; }
    }

    /**
     * Builder for creating FaceRecognitionService instances.
     */
    public static class Builder {
        private FaceDetector detector;
        private FeatureExtractor extractor;
        private FaceClassifier classifier;
        private Config config = new Config();

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

        public Builder config(Config config) {
            this.config = config;
            return this;
        }

        public FaceRecognitionService build() {
            Objects.requireNonNull(extractor, "Extractor is required");
            Objects.requireNonNull(classifier, "Classifier is required");
            return new FaceRecognitionService(detector, extractor, classifier, config);
        }
    }

    private FaceRecognitionService(FaceDetector detector, FeatureExtractor extractor,
                                   FaceClassifier classifier, Config config) {
        this.detector = detector;
        this.extractor = extractor;
        this.classifier = classifier;
        this.config = config;
        this.identities = new ConcurrentHashMap<>();
        this.trainingSamples = new ArrayList<>();
        this.trained = false;
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
     * Enrolls a face image for an identity.
     *
     * @param image the face image
     * @param identityName the identity name
     * @return the created or updated Identity
     */
    public Identity enroll(FaceImage image, String identityName) {
        return enroll(image, identityName, null);
    }

    /**
     * Enrolls a face image for an identity with external ID.
     *
     * @param image the face image
     * @param identityName the identity name
     * @param externalId optional external system ID
     * @return the created or updated Identity
     */
    public Identity enroll(FaceImage image, String identityName, String externalId) {
        Objects.requireNonNull(image, "Image cannot be null");
        Objects.requireNonNull(identityName, "Identity name cannot be null");

        // Find or create identity
        Identity identity = findIdentityByName(identityName);
        if (identity == null) {
            identity = new Identity(identityName);
            if (externalId != null) {
                identity.setExternalId(externalId);
            }
            identities.put(identity.getId(), identity);
        }

        // Add to training samples
        trainingSamples.add(new TrainingSample(image, identity.getId(), identityName));
        trained = false;

        return identity;
    }

    /**
     * Enrolls a face from a file.
     *
     * @param file the image file
     * @param identityName the identity name
     * @return the created or updated Identity
     * @throws IOException if the file cannot be read
     */
    public Identity enrollFromFile(File file, String identityName) throws IOException {
        FaceImage image = FaceImage.fromFile(file);
        return enroll(image, identityName);
    }

    /**
     * Trains the recognition system.
     * Must be called after enrolling faces and before recognition.
     */
    public void train() {
        if (trainingSamples.isEmpty()) {
            throw new IllegalStateException("No training samples enrolled");
        }

        // Prepare training data
        List<FaceImage> faces = new ArrayList<>();
        List<String> labels = new ArrayList<>();

        for (TrainingSample sample : trainingSamples) {
            // Optionally detect and crop face
            FaceImage processedImage = preprocessImage(sample.image);
            if (processedImage != null) {
                faces.add(processedImage);
                labels.add(sample.label);
            }
        }

        if (faces.isEmpty()) {
            throw new IllegalStateException("No valid faces found in training samples");
        }

        // Train extractor
        extractor.reset();
        extractor.train(faces, labels);

        // Extract features and enroll in classifier
        classifier.clear();

        Map<String, List<FeatureVector>> identityFeatures = new HashMap<>();
        for (int i = 0; i < faces.size(); i++) {
            FeatureVector features = extractor.extract(faces.get(i));
            String label = labels.get(i);

            identityFeatures.computeIfAbsent(label, k -> new ArrayList<>()).add(features);
        }

        // Create identity samples
        for (TrainingSample sample : trainingSamples) {
            Identity identity = identities.get(sample.identityId);
            if (identity != null) {
                List<FeatureVector> features = identityFeatures.get(sample.label);
                if (features != null && !features.isEmpty()) {
                    // Clear existing samples and add new ones
                    for (FeatureVector fv : features) {
                        identity.enrollSample(fv, 1.0, "training");
                    }
                    identityFeatures.remove(sample.label); // Only enroll once per identity
                }
            }
        }

        // Enroll identities in classifier
        for (Identity identity : identities.values()) {
            if (identity.hasSamples()) {
                classifier.enroll(identity);
            }
        }

        trained = true;
    }

    /**
     * Recognizes a face in an image.
     *
     * @param image the image to recognize
     * @return the recognition result
     */
    public RecognitionResult recognize(FaceImage image) {
        if (!trained) {
            throw new IllegalStateException("System not trained. Call train() first.");
        }

        long startTime = System.currentTimeMillis();
        long detectionTime = 0, extractionTime = 0, matchingTime = 0;

        // Preprocess image
        FaceImage processed = preprocessImage(image);
        if (processed == null) {
            return RecognitionResult.noFaceDetected();
        }

        detectionTime = System.currentTimeMillis() - startTime;
        long extractStart = System.currentTimeMillis();

        // Extract features
        FeatureVector features = extractor.extract(processed);

        extractionTime = System.currentTimeMillis() - extractStart;
        long matchStart = System.currentTimeMillis();

        // Classify
        RecognitionResult result = classifier.classify(features, config.getRecognitionThreshold());

        matchingTime = System.currentTimeMillis() - matchStart;
        long totalTime = System.currentTimeMillis() - startTime;

        // Add metrics to result
        RecognitionResult.ProcessingMetrics metrics = new RecognitionResult.ProcessingMetrics(
            detectionTime, extractionTime, matchingTime, totalTime);

        return RecognitionResult.builder()
            .status(result.getStatus())
            .bestMatch(result.getBestMatch().orElse(null))
            .alternatives(result.getAlternatives())
            .extractedFeatures(features)
            .metrics(metrics)
            .build();
    }

    /**
     * Recognizes a face from a file.
     *
     * @param file the image file
     * @return the recognition result
     * @throws IOException if the file cannot be read
     */
    public RecognitionResult recognizeFromFile(File file) throws IOException {
        FaceImage image = FaceImage.fromFile(file);
        return recognize(image);
    }

    /**
     * Preprocesses an image for recognition.
     */
    private FaceImage preprocessImage(FaceImage image) {
        // If detector is available, detect and crop face
        if (detector != null) {
            Optional<FaceRegion> face = detector.detectLargestFace(image);
            if (face.isEmpty()) {
                return null;
            }

            // Crop to face region
            FaceRegion region = face.get();
            BufferedImage cropped = image.getImage().getSubimage(
                Math.max(0, region.getX()),
                Math.max(0, region.getY()),
                Math.min(region.getWidth(), image.getWidth() - region.getX()),
                Math.min(region.getHeight(), image.getHeight() - region.getY())
            );
            image = FaceImage.fromBufferedImage(cropped);
        }

        // Resize to target dimensions
        if (image.getWidth() != config.getTargetWidth() ||
            image.getHeight() != config.getTargetHeight()) {
            image = image.resize(config.getTargetWidth(), config.getTargetHeight());
        }

        return image;
    }

    /**
     * Finds an identity by name.
     */
    private Identity findIdentityByName(String name) {
        for (Identity identity : identities.values()) {
            if (identity.getName().equals(name)) {
                return identity;
            }
        }
        return null;
    }

    /**
     * Gets all enrolled identities.
     *
     * @return collection of identities
     */
    public Collection<Identity> getIdentities() {
        return Collections.unmodifiableCollection(identities.values());
    }

    /**
     * Gets the number of enrolled identities.
     *
     * @return the identity count
     */
    public int getIdentityCount() {
        return identities.size();
    }

    /**
     * Checks if the system is trained.
     *
     * @return true if trained
     */
    public boolean isTrained() {
        return trained;
    }

    /**
     * Gets the feature extractor.
     *
     * @return the extractor
     */
    public FeatureExtractor getExtractor() {
        return extractor;
    }

    /**
     * Gets the classifier.
     *
     * @return the classifier
     */
    public FaceClassifier getClassifier() {
        return classifier;
    }

    /**
     * Gets the detector (may be null).
     *
     * @return the detector or null
     */
    public FaceDetector getDetector() {
        return detector;
    }

    /**
     * Gets the service configuration.
     *
     * @return the config
     */
    public Config getConfig() {
        return config;
    }

    @Override
    public String toString() {
        return String.format("FaceRecognitionService{extractor=%s, identities=%d, trained=%s}",
            extractor.getAlgorithmName(), identities.size(), trained);
    }
}
