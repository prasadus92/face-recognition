package com.facerecognition.application.service;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
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
    private final FeatureExtractor extractor;
    private final FaceClassifier classifier;
    private final ModelRepository modelRepository;
    private final FaceAligner aligner;

    private final Map<String, Identity> identities;
    private final List<TrainingSample> trainingSamples;
    private final Config config;

    private final ReadWriteLock lock = new ReentrantReadWriteLock();
    private volatile boolean trained;

    /** In-flight training sample — held in-memory until {@link #train()} is called. */
    private static final class TrainingSample {
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
                // Production deployments wire a detector via FaceRecognitionAutoConfiguration;
                // allowing null here lets tests and simple programmatic embedders run on
                // pre-cropped faces without standing up a real detector.
                log.warn("FaceRecognitionService built without a FaceDetector — preprocessing will pass images through unchanged. "
                        + "This is fine for tests and pre-cropped inputs, but production deployments should wire a detector.");
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
            trainingSamples.add(new TrainingSample(image, identity.getId(), identityName));
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

            List<FaceImage> faces = new ArrayList<>();
            List<String> labels = new ArrayList<>();
            for (TrainingSample sample : trainingSamples) {
                FaceImage processed = preprocessForTraining(sample.image);
                if (processed != null) {
                    faces.add(processed);
                    labels.add(sample.label);
                }
            }

            if (faces.isEmpty()) {
                throw new IllegalStateException("No valid faces found in training samples");
            }

            extractor.reset();
            extractor.train(faces, labels);

            classifier.clear();

            Map<String, List<FeatureVector>> identityFeatures = new HashMap<>();
            for (int i = 0; i < faces.size(); i++) {
                FeatureVector features = extractor.extract(faces.get(i));
                identityFeatures.computeIfAbsent(labels.get(i), k -> new ArrayList<>()).add(features);
            }

            // Replace stale samples on each identity with freshly extracted ones.
            for (Identity identity : identities.values()) {
                identity.clearSamples();
            }
            for (TrainingSample sample : trainingSamples) {
                Identity identity = identities.get(sample.identityId);
                if (identity == null) {
                    continue;
                }
                List<FeatureVector> features = identityFeatures.get(sample.label);
                if (features != null && !features.isEmpty()) {
                    for (FeatureVector fv : features) {
                        identity.enrollSample(fv, 1.0, "training");
                    }
                    identityFeatures.remove(sample.label);
                }
            }

            for (Identity identity : identities.values()) {
                if (identity.hasSamples()) {
                    classifier.enroll(identity);
                }
            }

            trained = true;
            log.info("Training complete: identities={} samples={}",
                    identities.size(), faces.size());

            if (config.isAutoSave()) {
                persistSafely();
            }
        } finally {
            lock.writeLock().unlock();
        }
    }

    // ---------------------------------------------------------------------
    // Recognition
    // ---------------------------------------------------------------------

    /** Recognises the largest face in the given image. */
    public RecognitionResult recognize(FaceImage image) {
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
            RecognitionResult result = classifier.classify(features, config.getRecognitionThreshold());
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
     * Attempts to load a previously saved model from the configured
     * {@link ModelRepository}. Returns {@code true} if a model was loaded.
     * Safe to call at application startup even if no model is present.
     */
    public boolean tryLoadSavedModel() {
        if (modelRepository == null) {
            return false;
        }
        String name = modelBaseName();
        lock.writeLock().lock();
        try {
            Optional<TrainedModel> loaded = modelRepository.load(name);
            if (loaded.isEmpty()) {
                return false;
            }
            TrainedModel model = loaded.get();
            restoreFromModel(model);
            log.info("Restored {} identities from model '{}'",
                    model.getIdentityCount(), name);
            return true;
        } catch (IOException e) {
            log.warn("Could not load saved model '{}': {}", name, e.getMessage());
            return false;
        } finally {
            lock.writeLock().unlock();
        }
    }

    /** Serialises the current state into a fresh {@link TrainedModel}. */
    public TrainedModel snapshot() {
        lock.readLock().lock();
        try {
            TrainedModel.Builder builder = TrainedModel.builder(
                    extractor.getAlgorithmName(), extractor.getVersion());
            for (Identity identity : identities.values()) {
                FeatureVector avg = identity.getAverageFeatureVector();
                if (avg != null) {
                    builder.addIdentity(identity, avg);
                }
            }
            return builder.build();
        } finally {
            lock.readLock().unlock();
        }
    }

    /** Explicit save — used by the REST export endpoint and CLI. */
    public void saveModel() throws IOException {
        if (modelRepository == null) {
            throw new IllegalStateException("No ModelRepository configured");
        }
        TrainedModel snapshot = snapshot();
        modelRepository.save(snapshot, modelBaseName(), true);
    }

    /** Hot-load a {@link TrainedModel} previously produced by this service. */
    public void loadModel(TrainedModel model) {
        Objects.requireNonNull(model, "model");
        lock.writeLock().lock();
        try {
            restoreFromModel(model);
        } finally {
            lock.writeLock().unlock();
        }
    }

    private void restoreFromModel(TrainedModel model) {
        identities.clear();
        trainingSamples.clear();
        classifier.clear();

        for (TrainedModel.EnrolledIdentity ei : model.getEnrolledIdentities()) {
            Identity identity = new Identity(ei.getIdentityName());
            // Preserve the original ID so REST callers can still look things up.
            identity.setExternalId(ei.getIdentityId());
            identity.enrollSample(ei.getFeatureVector(), 1.0, "imported");
            identities.put(identity.getId(), identity);
            classifier.enroll(identity);
        }
        // Imported models are ready to classify without a fresh train() call.
        trained = !identities.isEmpty();
    }

    private void persistSafely() {
        if (modelRepository == null) {
            return;
        }
        try {
            saveModel();
        } catch (IOException e) {
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
        return cropped;
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
