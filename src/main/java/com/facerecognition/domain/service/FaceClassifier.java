package com.facerecognition.domain.service;

import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;

import java.util.List;

/**
 * Interface for face classification services.
 * Implementations match feature vectors against enrolled identities
 * to perform face recognition.
 *
 * <p>Classification is the final step in the recognition pipeline,
 * determining which enrolled identity (if any) matches a probe face.</p>
 *
 * <p>Available implementations:</p>
 * <ul>
 *   <li><b>KNNClassifier</b>: k-Nearest Neighbors</li>
 *   <li><b>ThresholdClassifier</b>: Simple distance threshold</li>
 *   <li><b>SVMClassifier</b>: Support Vector Machine</li>
 *   <li><b>NeuralNetClassifier</b>: Neural network classifier</li>
 * </ul>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see FeatureVector
 * @see Identity
 */
public interface FaceClassifier {

    /**
     * Enrolls an identity with its feature vectors.
     *
     * @param identity the identity to enroll
     */
    void enroll(Identity identity);

    /**
     * Removes an enrolled identity.
     *
     * @param identityId the ID of the identity to remove
     * @return true if the identity was enrolled and removed
     */
    boolean unenroll(String identityId);

    /**
     * Checks if an identity is enrolled.
     *
     * @param identityId the identity ID
     * @return true if enrolled
     */
    boolean isEnrolled(String identityId);

    /**
     * Gets the number of enrolled identities.
     *
     * @return the count of enrolled identities
     */
    int getEnrolledCount();

    /**
     * Gets all enrolled identities.
     *
     * @return list of enrolled identities
     */
    List<Identity> getEnrolledIdentities();

    /**
     * Classifies a feature vector against enrolled identities.
     *
     * @param probe the feature vector to classify
     * @return the recognition result
     */
    RecognitionResult classify(FeatureVector probe);

    /**
     * Classifies with a custom confidence threshold.
     *
     * @param probe the feature vector to classify
     * @param threshold the minimum confidence for a match
     * @return the recognition result
     */
    RecognitionResult classify(FeatureVector probe, double threshold);

    /**
     * Gets the top N matches for a feature vector.
     *
     * @param probe the feature vector
     * @param n the number of matches to return
     * @return recognition result with top N alternatives
     */
    RecognitionResult getTopMatches(FeatureVector probe, int n);

    /**
     * Computes the distance between a probe and an enrolled identity.
     *
     * @param probe the probe feature vector
     * @param identityId the enrolled identity ID
     * @return the distance, or Double.MAX_VALUE if not enrolled
     */
    double getDistance(FeatureVector probe, String identityId);

    /**
     * Gets the name of this classifier implementation.
     *
     * @return the classifier name
     */
    String getName();

    /**
     * Gets the distance metric used by this classifier.
     *
     * @return the distance metric
     */
    DistanceMetric getDistanceMetric();

    /**
     * Sets the distance metric to use.
     *
     * @param metric the distance metric
     */
    void setDistanceMetric(DistanceMetric metric);

    /**
     * Clears all enrolled identities.
     */
    void clear();

    /**
     * Retrains the classifier (for classifiers that need training).
     */
    void retrain();

    /**
     * Available distance metrics for face comparison.
     */
    enum DistanceMetric {
        /** Euclidean (L2) distance. */
        EUCLIDEAN,
        /** Cosine distance (1 - cosine similarity). */
        COSINE,
        /** Manhattan (L1) distance. */
        MANHATTAN,
        /** Chi-square distance (for histograms). */
        CHI_SQUARE,
        /** Mahalanobis distance. */
        MAHALANOBIS
    }

    /**
     * Classifier configuration options.
     */
    class ClassifierConfig {
        private double threshold = 0.6;
        private int k = 1;
        private DistanceMetric metric = DistanceMetric.EUCLIDEAN;
        private boolean useAverageFeatures = false;

        public double getThreshold() { return threshold; }
        public ClassifierConfig setThreshold(double t) { this.threshold = t; return this; }

        public int getK() { return k; }
        public ClassifierConfig setK(int k) { this.k = k; return this; }

        public DistanceMetric getMetric() { return metric; }
        public ClassifierConfig setMetric(DistanceMetric m) { this.metric = m; return this; }

        public boolean isUseAverageFeatures() { return useAverageFeatures; }
        public ClassifierConfig setUseAverageFeatures(boolean u) { this.useAverageFeatures = u; return this; }
    }
}
