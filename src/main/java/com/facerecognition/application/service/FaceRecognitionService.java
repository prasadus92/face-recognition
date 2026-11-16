package com.facerecognition.application.service;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FaceRegion;
import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.domain.service.FaceClassifier;
import com.facerecognition.domain.service.FaceDetector;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.infrastructure.persistence.InvalidModelException;
import com.facerecognition.infrastructure.persistence.ModelCodec;
import com.facerecognition.infrastructure.persistence.ModelRepository;
import com.facerecognition.infrastructure.persistence.TrainedModel;
import com.facerecognition.infrastructure.preprocessing.FaceAligner;

/**
 * Main application service orchestrating the face-recognition pipeline.
 *
 * <p>The pipeline is:</p>
 * <ol>
 *   <li>{@link FaceDetector} — locate the largest face in the input image</li>
 *   <li>{@link FaceAligner} — (optional) eye-align and histogram-equalise the crop</li>
 *   <li>{@link FeatureExtractor} — extract a {@link FeatureVector}</li>
 *   <li>{@link FaceClassifier} — match against enrolled identities</li>
 * </ol>
 *
 * <p><b>Thread-safety:</b> all public mutation and classification entry points
 * are guarded by a {@link ReentrantReadWriteLock}. Multiple concurrent
 * {@link #recognize(FaceImage) recognize} calls are served in parallel; training
 * and enrolment take the write lock so they cannot race with either readers or
 * one another.</p>
 *
 * <p><b>Detector required.</b> The service refuses to build without a
 * {@link FaceDetector}. The previous version silently bypassed detection when
 * none was configured, which made every recognition run against the full image
 * — this has been fixed.</p>
 */
public class FaceRecognitionService {

    private static final Logger log = LoggerFactory.getLogger(FaceRecognitionService.class);

    private final FaceDetector detector;
    private volatile FeatureExtractor extractor;
    private final FaceClassifier classifier;
    private final ModelRepository modelRepository;
    private final FaceAligner aligner;

    private final Map<String, Identity> identities;
    private final List<TrainingSample> trainingSamples;
    private final Config config;

    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    private volatile boolean trained;

    /** A preprocessed grayscale crop kept for retraining and persisted with the model. */
    private record TrainingSample(String identityId, FaceImage crop) {
    }

    /**
     * Runtime configuration for the service. Typically populated from
     * {@code application.yml} via {@code FaceRecognitionAutoConfiguration},
     * but may be constructed manually in tests or programmatic embedders.
     */
    public static class Config {
        private double recognitionThreshold = 0.6;
        private double detectionConfidence = 0.5;
        private double minQuality = 0.3;
        private boolean autoAlign = true;
        private boolean histogramEqualization = true;
        private int targetWidth = 100;
        private int targetHeight = 100;
        private boolean autoSave = false;
        private boolean autoLoad = false;
        private String modelFileName = "default.frm";

        public double getRecognitionThreshold() { return recognitionThreshold; }
        public Config setRecognitionThreshold(double t) { this.recognitionThreshold = t; return this; }

        public double getDetectionConfidence() { return detectionConfidence; }
        public Config setDetectionConfidence(double c) { this.detectionConfidence = c; return this; }

        public double getMinQuality() { return minQuality; }
        public Config setMinQuality(double q) { this.minQuality = q; return this; }

        public boolean isAutoAlign() { return autoAlign; }
        public Config setAutoAlign(boolean a) { this.autoAlign = a; return this; }

        public boolean isHistogramEqualization() { return histogramEqualization; }
        public Config setHistogramEqualization(boolean h) { this.histogramEqualization = h; return this; }

        public int getTargetWidth() { return targetWidth; }
        public Config setTargetWidth(int w) { this.targetWidth = w; return this; }

        public int getTargetHeight() { return targetHeight; }
        public Config setTargetHeight(int h) { this.targetHeight = h; return this; }

        public boolean isAutoSave() { return autoSave; }
        public Config setAutoSave(boolean v) { this.autoSave = v; return this; }

        public boolean isAutoLoad() { return autoLoad; }
        public Config setAutoLoad(boolean v) { this.autoLoad = v; return this; }

        public String getModelFileName() { return modelFileName; }
        public Config setModelFileName(String name) { this.modelFileName = name; return this; }
    }

    /** Fluent builder. Detector, extractor and classifier are all required. */
    public static class Builder {
        private FaceDetector detector;
        private FeatureExtractor extractor;
        private FaceClassifier classifier;
        private ModelRepository modelRepository;
        private Config config = new Config();

        public Builder detector(FaceDetector detector) { this.detector = detector; return this; }
        public Builder extractor(FeatureExtractor extractor) { this.extractor = extractor; return this; }
        public Builder classifier(FaceClassifier classifier) { this.classifier = classifier; return this; }
        public Builder modelRepository(ModelRepository repo) { this.modelRepository = repo; return this; }
        public Builder config(Config config) { this.config = config != null ? config : new Config(); return this; }

        public FaceRecognitionService build() {
            Objects.requireNonNull(extractor, "Extractor is required");
            Objects.requireNonNull(classifier, "Classifier is required");
            if (detector == null) {
                log.info("No face detector configured; images are used as face crops");
            }
            return new FaceRecognitionService(detector, extractor, classifier, modelRepository, config);
        }
    }

    public static Builder builder() {
        return new Builder();
    }

    private FaceRecognitionService(FaceDetector detector,
                                   FeatureExtractor extractor,
                                   FaceClassifier classifier,
                                   ModelRepository modelRepository,
                                   Config config) {
        this.detector = detector;
        this.extractor = extractor;
        this.classifier = classifier;
        this.modelRepository = modelRepository;
        this.config = config;
        this.identities = new ConcurrentHashMap<>();
        this.trainingSamples = new ArrayList<>();
        this.aligner = new FaceAligner.Builder()
                .targetSize(config.getTargetWidth(), config.getTargetHeight())
                .histogramEqualization(config.isHistogramEqualization())
                .build();
        this.trained = false;
    }

    // ---------------------------------------------------------------------
    // Enrolment
    // ---------------------------------------------------------------------

    /** Enrol a face image under the given name. */
    public Identity enroll(FaceImage image, String identityName) {
        return enroll(image, identityName, null);
    }

    /** Enrol a face image with an external identifier (e.g. employee ID). */
    public Identity enroll(FaceImage image, String identityName, String externalId) {
        Objects.requireNonNull(image, "Image cannot be null");
        Objects.requireNonNull(identityName, "Identity name cannot be null");
        FaceImage crop = preprocessForTraining(image);

        lock.writeLock().lock();
        try {
            Identity identity = findIdentityByNameInternal(identityName);
            if (identity == null) {
                identity = new Identity(identityName);
                if (externalId != null) {
                    identity.setExternalId(externalId);
                }
                identities.put(identity.getId(), identity);
            }
            trainingSamples.add(new TrainingSample(identity.getId(), crop));
            trained = false;
            return identity;
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** Convenience overload that reads the image from a file. */
    public Identity enrollFromFile(File file, String identityName) throws IOException {
        return enroll(FaceImage.fromFile(file), identityName);
    }

    // ---------------------------------------------------------------------
    // Training
    // ---------------------------------------------------------------------

    /**
     * Trains the extractor and classifier on all enrolled samples.
     *
     * <p>If {@code facerecognition.model.auto-save=true} the trained model is
     * persisted via the configured {@link ModelRepository} after a successful
     * run. Persistence failures are logged but do not fail the training call.</p>
     */
    public void train() {
        lock.writeLock().lock();
        try {
            if (trainingSamples.isEmpty()) {
                throw new IllegalStateException("No training samples enrolled");
            }
            List<FaceImage> faces = trainingSamples.stream().map(TrainingSample::crop).toList();
            List<String> labels = trainingSamples.stream().map(TrainingSample::identityId).toList();

            extractor.reset();
            extractor.train(faces, labels);
            applyGallery(extractGallery(extractor, trainingSamples));

            log.info("Training complete: identities={} samples={}", identities.size(), faces.size());
            if (config.isAutoSave()) {
                persistSafely();
            }
        } finally {
            lock.writeLock().unlock();
        }
    }

    private static Map<String, List<FeatureVector>> extractGallery(FeatureExtractor fittedExtractor,
                                                                  List<TrainingSample> samples) {
        Map<String, List<FeatureVector>> gallery = new HashMap<>();
        for (TrainingSample sample : samples) {
            gallery.computeIfAbsent(sample.identityId(), id -> new ArrayList<>())
                    .add(fittedExtractor.extract(sample.crop()));
        }
        return gallery;
    }

    private void applyGallery(Map<String, List<FeatureVector>> gallery) {
        classifier.clear();
        for (Identity identity : identities.values()) {
            identity.clearSamples();
            gallery.getOrDefault(identity.getId(), List.of())
                    .forEach(features -> identity.enrollSample(features, 1.0, "training"));
            if (identity.hasSamples()) {
                classifier.enroll(identity);
            }
        }
        trained = true;
    }

    /** Recognizes the largest face in the image using the configured threshold. */
    public RecognitionResult recognize(FaceImage image) {
        return recognize(image, config.getRecognitionThreshold());
    }

    /**
     * Recognizes the largest face in the image.
     *
     * @param image     the image to search
     * @param threshold minimum confidence for a match, 0 to 1
     * @return the match, or an unknown or no-face result
     */
    public RecognitionResult recognize(FaceImage image, double threshold) {
        lock.readLock().lock();
        try {
            if (!trained) {
                throw new IllegalStateException("System not trained. Call train() first.");
            }

            long start = System.currentTimeMillis();

            FaceImage processed = preprocessForRecognition(image);
            if (processed == null) {
                return RecognitionResult.noFaceDetected();
            }
            long detectionTime = System.currentTimeMillis() - start;

            long extractStart = System.currentTimeMillis();
            FeatureVector features = extractor.extract(processed);
            long extractionTime = System.currentTimeMillis() - extractStart;

            long matchStart = System.currentTimeMillis();
            RecognitionResult result = classifier.classify(features, threshold);
            long matchingTime = System.currentTimeMillis() - matchStart;

            long totalTime = System.currentTimeMillis() - start;

            RecognitionResult.ProcessingMetrics metrics = new RecognitionResult.ProcessingMetrics(
                    detectionTime, extractionTime, matchingTime, totalTime);

            return RecognitionResult.builder()
                    .status(result.getStatus())
                    .bestMatch(result.getBestMatch().orElse(null))
                    .alternatives(result.getAlternatives())
                    .extractedFeatures(features)
                    .metrics(metrics)
                    .build();
        } finally {
            lock.readLock().unlock();
        }
    }

    /** Convenience overload reading from a file. */
    public RecognitionResult recognizeFromFile(File file) throws IOException {
        return recognize(FaceImage.fromFile(file));
    }

    // ---------------------------------------------------------------------
    // Model persistence
    // ---------------------------------------------------------------------

    /**
     * Restores the model saved under the configured name, if there is one.
     *
     * @return {@code true} if a model was loaded
     */
    public boolean tryLoadSavedModel() {
        if (modelRepository == null) {
            return false;
        }
        String name = modelBaseName();
        try {
            Optional<byte[]> data = modelRepository.load(name);
            if (data.isEmpty()) {
                return false;
            }
            importModel(data.get());
            log.info("Restored {} identities from model '{}'", identities.size(), name);
            return true;
        } catch (IOException e) {
            log.warn("Could not load saved model '{}': {}", name, e.getMessage());
            return false;
        }
    }

    /**
     * Encodes the current extractor, identities and training crops (see {@link ModelCodec}).
     *
     * @return the encoded model
     */
    public byte[] exportModel() {
        lock.readLock().lock();
        try {
            return ModelCodec.encode(toModel());
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * Replaces all state with a model produced by {@link #exportModel()}. The model's extractor
     * replaces the configured one. On failure the current state is left untouched.
     *
     * @param data the encoded model
     * @throws InvalidModelException if the data is not a valid model
     */
    public void importModel(byte[] data) throws InvalidModelException {
        TrainedModel model = ModelCodec.decode(data);
        lock.writeLock().lock();
        try {
            restore(model);
        } catch (RuntimeException e) {
            throw new InvalidModelException("Model file could not be restored", e);
        } finally {
            lock.writeLock().unlock();
        }
        log.info("Imported a {} model with {} identities", extractor.getAlgorithmName(), identities.size());
    }

    /**
     * Writes the current model to the configured {@link ModelRepository}.
     *
     * @throws IOException if the model cannot be written
     */
    public void saveModel() throws IOException {
        if (modelRepository == null) {
            throw new IllegalStateException("No ModelRepository configured");
        }
        modelRepository.save(modelBaseName(), exportModel());
    }

    private TrainedModel toModel() {
        List<TrainedModel.IdentityRecord> records = identities.values().stream()
                .map(i -> new TrainedModel.IdentityRecord(
                        i.getId(), i.getName(), i.getExternalId(), i.isActive(), i.getAllMetadata()))
                .toList();
        List<TrainedModel.FaceSample> samples = trainingSamples.stream()
                .map(s -> new TrainedModel.FaceSample(
                        s.identityId(), s.crop().getWidth(), s.crop().getHeight(), s.crop().toGrayscaleBytes()))
                .toList();
        return new TrainedModel(extractor, trained, records, samples, Instant.now());
    }

    private void restore(TrainedModel model) {
        Map<String, Identity> restoredIdentities = new LinkedHashMap<>();
        for (TrainedModel.IdentityRecord record : model.identities()) {
            Identity identity = new Identity(record.id(), record.name(), record.externalId());
            identity.setActive(record.active());
            record.metadata().forEach(identity::setMetadata);
            restoredIdentities.put(identity.getId(), identity);
        }
        List<TrainingSample> restoredSamples = model.samples().stream()
                .map(sample -> new TrainingSample(sample.identityId(),
                        FaceImage.fromGrayscale(sample.width(), sample.height(), sample.pixels())))
                .toList();
        Map<String, List<FeatureVector>> gallery = model.trained()
                ? extractGallery(model.extractor(), restoredSamples)
                : Map.of();

        identities.clear();
        identities.putAll(restoredIdentities);
        trainingSamples.clear();
        trainingSamples.addAll(restoredSamples);
        extractor = model.extractor();
        if (model.trained()) {
            applyGallery(gallery);
        } else {
            classifier.clear();
            trained = false;
        }
    }

    private void persistSafely() {
        if (modelRepository == null) {
            return;
        }
        try {
            saveModel();
        } catch (IOException | IllegalStateException e) {
            log.warn("Auto-save failed: {}", e.getMessage());
        }
    }

    private String modelBaseName() {
        String name = config.getModelFileName();
        if (name == null || name.isBlank()) {
            return "default";
        }
        int dot = name.lastIndexOf('.');
        return dot > 0 ? name.substring(0, dot) : name;
    }

    // ---------------------------------------------------------------------
    // Preprocessing
    // ---------------------------------------------------------------------

    /**
     * Preprocessing pipeline for enrolment and training samples:
     * detect → align/crop → resize → normalise.
     */
    private FaceImage preprocessForTraining(FaceImage image) {
        return preprocessInternal(image, true);
    }

    /** Preprocessing pipeline for recognition requests. */
    private FaceImage preprocessForRecognition(FaceImage image) {
        return preprocessInternal(image, false);
    }

    private FaceImage preprocessInternal(FaceImage image, boolean skipIfUndetectable) {
        FaceImage cropped = image;

        if (detector != null) {
            // 1. Face detection.
            Optional<FaceRegion> faceOpt = detector.detectLargestFace(image);
            if (faceOpt.isPresent()) {
                FaceRegion region = faceOpt.get();
                cropped = cropToRegion(image, region);

                // 2. Optional alignment using detected landmarks.
                if (config.isAutoAlign()) {
                    if (region.hasLandmarks()) {
                        cropped = aligner.align(image, region.getLandmarks());
                    } else {
                        cropped = aligner.alignFromRegion(image, region);
                    }
                }
            } else if (skipIfUndetectable) {
                // During training we tolerate detection failures and fall back to
                // the whole image — this lets users enrol from pre-cropped faces.
                cropped = image;
            } else {
                return null;
            }
        }
        // If no detector is configured, pass the image through untouched
        // and rely on the caller to have pre-cropped it.

        // 3. Resize to the target size expected by the extractor.
        if (cropped.getWidth() != config.getTargetWidth()
                || cropped.getHeight() != config.getTargetHeight()) {
            cropped = cropped.resize(config.getTargetWidth(), config.getTargetHeight());
        }
        return cropped.toGrayscale();
    }

    private FaceImage cropToRegion(FaceImage image, FaceRegion region) {
        int x = Math.max(0, region.getX());
        int y = Math.max(0, region.getY());
        int w = Math.min(region.getWidth(), image.getWidth() - x);
        int h = Math.min(region.getHeight(), image.getHeight() - y);
        if (w <= 0 || h <= 0) {
            return image;
        }
        BufferedImage sub = image.getImage().getSubimage(x, y, w, h);
        return FaceImage.fromBufferedImage(sub);
    }

    // ---------------------------------------------------------------------
    // Read access
    // ---------------------------------------------------------------------

    private Identity findIdentityByNameInternal(String name) {
        for (Identity identity : identities.values()) {
            if (identity.getName().equals(name)) {
                return identity;
            }
        }
        return null;
    }

    /** @return an unmodifiable snapshot of all enrolled identities. */
    public Collection<Identity> getIdentities() {
        lock.readLock().lock();
        try {
            return Collections.unmodifiableCollection(new ArrayList<>(identities.values()));
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * @param identityId the identity to count for
     * @return how many face images are enrolled for it, whether or not the model is trained yet
     */
    public int getEnrolledImageCount(String identityId) {
        lock.readLock().lock();
        try {
            return (int) trainingSamples.stream().filter(s -> s.identityId().equals(identityId)).count();
        } finally {
            lock.readLock().unlock();
        }
    }

    /** @return how many face images are enrolled in total, whether or not the model is trained yet */
    public int getEnrolledImageCount() {
        lock.readLock().lock();
        try {
            return trainingSamples.size();
        } finally {
            lock.readLock().unlock();
        }
    }

    public int getIdentityCount() {
        return identities.size();
    }

    public boolean isTrained() {
        return trained;
    }

    public FeatureExtractor getExtractor() {
        return extractor;
    }

    public FaceClassifier getClassifier() {
        return classifier;
    }

    public FaceDetector getDetector() {
        return detector;
    }

    public Config getConfig() {
        return config;
    }

    public ModelRepository getModelRepository() {
        return modelRepository;
    }

    @Override
    public String toString() {
        return String.format("FaceRecognitionService{extractor=%s, identities=%d, trained=%s}",
                extractor.getAlgorithmName(), identities.size(), trained);
    }
}
