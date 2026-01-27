package com.facerecognition.domain.service;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FeatureVector;

import java.util.List;
import java.util.stream.Collectors;

/**
 * Interface for face feature extraction services.
 * Implementations convert face images into numerical feature
 * vectors that can be compared for recognition.
 *
 * <p>Feature extraction is the core of face recognition. Different
 * algorithms produce different types of feature vectors:</p>
 *
 * <ul>
 *   <li><b>Eigenfaces (PCA)</b>: Projects faces onto eigenface basis</li>
 *   <li><b>Fisherfaces (LDA)</b>: Maximizes class separability</li>
 *   <li><b>LBPH</b>: Local Binary Pattern Histograms</li>
 *   <li><b>Deep Learning</b>: CNN-based embeddings (FaceNet, ArcFace)</li>
 * </ul>
 *
 * <p>Extractors must be trained on a dataset before use. Training
 * learns the projection matrices or model parameters needed for
 * feature extraction.</p>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see FeatureVector
 */
public interface FeatureExtractor {

    /**
     * Trains the feature extractor on a dataset.
     *
     * @param faces list of face images for training
     * @param labels corresponding identity labels (may be null for unsupervised)
     * @throws IllegalStateException if already trained
     */
    void train(List<FaceImage> faces, List<String> labels);

    /**
     * Checks if the extractor has been trained.
     *
     * @return true if trained and ready for extraction
     */
    boolean isTrained();

    /**
     * Extracts features from a face image.
     *
     * @param face the face image (should be aligned/normalized)
     * @return the extracted feature vector
     * @throws IllegalStateException if not trained
     */
    FeatureVector extract(FaceImage face);

    /**
     * Extracts features from multiple face images.
     *
     * @param faces the face images
     * @return list of feature vectors in the same order
     */
    default List<FeatureVector> extractBatch(List<FaceImage> faces) {
        return faces.stream()
            .map(this::extract)
            .collect(Collectors.toList());
    }

    /**
     * Gets the dimension of output feature vectors.
     *
     * @return the feature vector dimension
     */
    int getFeatureDimension();

    /**
     * Gets the name of this extraction algorithm.
     *
     * @return the algorithm name
     */
    String getAlgorithmName();

    /**
     * Gets the version of this extractor.
     *
     * @return the version number
     */
    int getVersion();

    /**
     * Gets the expected input image size.
     *
     * @return array of [width, height]
     */
    int[] getExpectedImageSize();

    /**
     * Resets the extractor to untrained state.
     */
    void reset();

    /**
     * Gets configuration information about this extractor.
     *
     * @return the configuration
     */
    ExtractorConfig getConfig();

    /**
     * Configuration for feature extractors.
     */
    class ExtractorConfig {
        private int numComponents = 10;
        private boolean normalize = true;
        private int imageWidth = 48;
        private int imageHeight = 64;

        public int getNumComponents() { return numComponents; }
        public ExtractorConfig setNumComponents(int n) { this.numComponents = n; return this; }

        public boolean isNormalize() { return normalize; }
        public ExtractorConfig setNormalize(boolean n) { this.normalize = n; return this; }

        public int getImageWidth() { return imageWidth; }
        public ExtractorConfig setImageWidth(int w) { this.imageWidth = w; return this; }

        public int getImageHeight() { return imageHeight; }
        public ExtractorConfig setImageHeight(int h) { this.imageHeight = h; return this; }
    }
}
