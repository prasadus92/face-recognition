package com.facerecognition.domain.model;

import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

/**
 * Represents a feature vector extracted from a face image.
 * Feature vectors are numerical representations of faces used
 * for comparison and classification.
 *
 * <p>Feature vectors can be compared using various distance metrics.
 * The dimension of the vector depends on the extraction algorithm:</p>
 * <ul>
 *   <li>Eigenfaces: typically 10-200 dimensions</li>
 *   <li>Fisherfaces: typically num_classes - 1 dimensions</li>
 *   <li>LBPH: typically 256 * num_regions dimensions</li>
 *   <li>Deep learning: typically 128-512 dimensions</li>
 * </ul>
 *
 * @author Face Recognition Team
 * @version 2.0
 * @since 2.0
 * @see com.facerecognition.domain.service.FeatureExtractor
 */
public class FeatureVector implements Serializable {

    private static final long serialVersionUID = 2L;

    private final double[] features;
    private final String algorithmName;
    private final int algorithmVersion;

    // Cached norm for faster comparisons
    private transient Double cachedNorm;

    /**
     * Creates a new feature vector.
     *
     * @param features the feature values
     * @param algorithmName the name of the extraction algorithm
     * @param algorithmVersion the algorithm version
     * @throws IllegalArgumentException if features is null or empty
     */
    public FeatureVector(double[] features, String algorithmName, int algorithmVersion) {
        Objects.requireNonNull(features, "Features array cannot be null");
        if (features.length == 0) {
            throw new IllegalArgumentException("Features array cannot be empty");
        }

        this.features = Arrays.copyOf(features, features.length);
        this.algorithmName = algorithmName != null ? algorithmName : "unknown";
        this.algorithmVersion = algorithmVersion;
    }

    /**
     * Creates a feature vector with default algorithm info.
     *
     * @param features the feature values
     */
    public FeatureVector(double[] features) {
        this(features, "unknown", 1);
    }

    /**
     * Gets the feature values array.
     *
     * @return a copy of the feature values
     */
    public double[] getFeatures() {
        return Arrays.copyOf(features, features.length);
    }

    /**
     * Gets a specific feature value by index.
     *
     * @param index the feature index
     * @return the feature value
     * @throws IndexOutOfBoundsException if index is out of range
     */
    public double getFeature(int index) {
        return features[index];
    }

    /**
     * Gets the dimension (length) of the feature vector.
     *
     * @return the number of features
     */
    public int getDimension() {
        return features.length;
    }

    /**
     * Gets the name of the algorithm used to extract these features.
     *
     * @return the algorithm name
     */
    public String getAlgorithmName() {
        return algorithmName;
    }

    /**
     * Gets the version of the extraction algorithm.
     *
     * @return the algorithm version
     */
    public int getAlgorithmVersion() {
        return algorithmVersion;
    }

    /**
     * Computes the Euclidean (L2) norm of this vector.
     *
     * @return the L2 norm
     */
    public double norm() {
        if (cachedNorm == null) {
            double sum = 0;
            for (double f : features) {
                sum += f * f;
            }
            cachedNorm = Math.sqrt(sum);
        }
        return cachedNorm;
    }

    /**
     * Computes the Euclidean distance to another feature vector.
     *
     * @param other the other feature vector
     * @return the Euclidean distance
     * @throws IllegalArgumentException if dimensions don't match
     */
    public double euclideanDistance(FeatureVector other) {
        validateDimensions(other);

        double sum = 0;
        for (int i = 0; i < features.length; i++) {
            double diff = features[i] - other.features[i];
            sum += diff * diff;
        }
        return Math.sqrt(sum);
    }

    /**
     * Computes the cosine similarity with another feature vector.
     * Returns a value between -1 (opposite) and 1 (identical).
     *
     * @param other the other feature vector
     * @return the cosine similarity
     * @throws IllegalArgumentException if dimensions don't match
     */
    public double cosineSimilarity(FeatureVector other) {
        validateDimensions(other);

        double dotProduct = 0;
        for (int i = 0; i < features.length; i++) {
            dotProduct += features[i] * other.features[i];
        }
        return dotProduct / (this.norm() * other.norm());
    }

    /**
     * Computes the cosine distance with another feature vector.
     * Returns a value between 0 (identical) and 2 (opposite).
     *
     * @param other the other feature vector
     * @return the cosine distance
     */
    public double cosineDistance(FeatureVector other) {
        return 1.0 - cosineSimilarity(other);
    }

    /**
     * Computes the Manhattan (L1) distance to another feature vector.
     *
     * @param other the other feature vector
     * @return the Manhattan distance
     */
    public double manhattanDistance(FeatureVector other) {
        validateDimensions(other);

        double sum = 0;
        for (int i = 0; i < features.length; i++) {
            sum += Math.abs(features[i] - other.features[i]);
        }
        return sum;
    }

    /**
     * Computes the Chi-square distance to another feature vector.
     * Commonly used for histogram-based features like LBPH.
     *
     * @param other the other feature vector
     * @return the Chi-square distance
     */
    public double chiSquareDistance(FeatureVector other) {
        validateDimensions(other);

        double sum = 0;
        for (int i = 0; i < features.length; i++) {
            double diff = features[i] - other.features[i];
            double total = features[i] + other.features[i];
            if (total > 0) {
                sum += (diff * diff) / total;
            }
        }
        return sum;
    }

    /**
     * Returns a normalized (unit length) version of this vector.
     *
     * @return a new normalized FeatureVector
     */
    public FeatureVector normalize() {
        double n = norm();
        if (n == 0) {
            return this;
        }

        double[] normalized = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            normalized[i] = features[i] / n;
        }
        return new FeatureVector(normalized, algorithmName, algorithmVersion);
    }

    /**
     * Adds another feature vector to this one (element-wise).
     *
     * @param other the vector to add
     * @return a new FeatureVector with summed values
     */
    public FeatureVector add(FeatureVector other) {
        validateDimensions(other);

        double[] result = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            result[i] = features[i] + other.features[i];
        }
        return new FeatureVector(result, algorithmName, algorithmVersion);
    }

    /**
     * Subtracts another feature vector from this one.
     *
     * @param other the vector to subtract
     * @return a new FeatureVector with the difference
     */
    public FeatureVector subtract(FeatureVector other) {
        validateDimensions(other);

        double[] result = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            result[i] = features[i] - other.features[i];
        }
        return new FeatureVector(result, algorithmName, algorithmVersion);
    }

    /**
     * Multiplies this vector by a scalar.
     *
     * @param scalar the scalar value
     * @return a new scaled FeatureVector
     */
    public FeatureVector scale(double scalar) {
        double[] result = new double[features.length];
        for (int i = 0; i < features.length; i++) {
            result[i] = features[i] * scalar;
        }
        return new FeatureVector(result, algorithmName, algorithmVersion);
    }

    /**
     * Computes the dot product with another vector.
     *
     * @param other the other vector
     * @return the dot product
     */
    public double dot(FeatureVector other) {
        validateDimensions(other);

        double result = 0;
        for (int i = 0; i < features.length; i++) {
            result += features[i] * other.features[i];
        }
        return result;
    }

    private void validateDimensions(FeatureVector other) {
        Objects.requireNonNull(other, "Other feature vector cannot be null");
        if (features.length != other.features.length) {
            throw new IllegalArgumentException(
                String.format("Dimension mismatch: %d vs %d", features.length, other.features.length));
        }
    }

    /**
     * Checks if this vector is compatible with another for comparison.
     *
     * @param other the other feature vector
     * @return true if vectors can be compared
     */
    public boolean isCompatibleWith(FeatureVector other) {
        return other != null &&
               features.length == other.features.length &&
               algorithmName.equals(other.algorithmName);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        FeatureVector that = (FeatureVector) o;
        return Arrays.equals(features, that.features);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(features);
    }

    @Override
    public String toString() {
        return String.format("FeatureVector{dim=%d, algorithm=%s, norm=%.4f}",
            features.length, algorithmName, norm());
    }

    /**
     * Returns a string representation showing the first few feature values.
     *
     * @param maxValues maximum number of values to show
     * @return a string with feature values
     */
    public String toDetailedString(int maxValues) {
        StringBuilder sb = new StringBuilder();
        sb.append("FeatureVector[");
        int count = Math.min(maxValues, features.length);
        for (int i = 0; i < count; i++) {
            if (i > 0) sb.append(", ");
            sb.append(String.format("%.4f", features[i]));
        }
        if (count < features.length) {
            sb.append(", ... (").append(features.length - count).append(" more)");
        }
        sb.append("]");
        return sb.toString();
    }
}
