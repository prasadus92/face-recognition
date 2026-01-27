package com.facerecognition.domain.model;

import java.io.Serializable;
import java.time.LocalDateTime;
import java.util.*;

/**
 * Represents a known identity in the face recognition system.
 * An identity can have multiple enrolled face samples, which are
 * averaged or compared individually depending on the algorithm.
 *
 * <p>Identities are the core unit of recognition - the system
 * classifies unknown faces by matching them to enrolled identities.</p>
 *
 * @author Face Recognition Team
 * @version 2.0
 * @since 2.0
 */
public class Identity implements Serializable {

    private static final long serialVersionUID = 2L;

    private final String id;
    private String name;
    private String externalId;
    private final LocalDateTime createdAt;
    private LocalDateTime updatedAt;
    private final List<EnrolledSample> samples;
    private final Map<String, String> metadata;
    private boolean active;

    /**
     * Represents a single enrolled face sample for an identity.
     */
    public static class EnrolledSample implements Serializable {
        private static final long serialVersionUID = 1L;

        private final String sampleId;
        private final FeatureVector features;
        private final LocalDateTime enrolledAt;
        private final double qualityScore;
        private final String sourceDescription;

        /**
         * Creates a new enrolled sample.
         *
         * @param features the extracted feature vector
         * @param qualityScore the face quality score
         * @param sourceDescription description of the source image
         */
        public EnrolledSample(FeatureVector features, double qualityScore, String sourceDescription) {
            this.sampleId = UUID.randomUUID().toString();
            this.features = Objects.requireNonNull(features);
            this.enrolledAt = LocalDateTime.now();
            this.qualityScore = qualityScore;
            this.sourceDescription = sourceDescription != null ? sourceDescription : "";
        }

        public String getSampleId() { return sampleId; }
        public FeatureVector getFeatures() { return features; }
        public LocalDateTime getEnrolledAt() { return enrolledAt; }
        public double getQualityScore() { return qualityScore; }
        public String getSourceDescription() { return sourceDescription; }

        @Override
        public String toString() {
            return String.format("EnrolledSample{id=%s, quality=%.2f, enrolledAt=%s}",
                sampleId.substring(0, 8), qualityScore, enrolledAt);
        }
    }

    /**
     * Creates a new Identity with the specified name.
     *
     * @param name the identity name
     */
    public Identity(String name) {
        this(UUID.randomUUID().toString(), name, null);
    }

    /**
     * Creates a new Identity with a specific ID and name.
     *
     * @param id the unique identifier
     * @param name the identity name
     * @param externalId an optional external system ID
     */
    public Identity(String id, String name, String externalId) {
        this.id = Objects.requireNonNull(id, "ID cannot be null");
        this.name = Objects.requireNonNull(name, "Name cannot be null");
        this.externalId = externalId;
        this.createdAt = LocalDateTime.now();
        this.updatedAt = this.createdAt;
        this.samples = new ArrayList<>();
        this.metadata = new HashMap<>();
        this.active = true;
    }

    /**
     * Gets the unique identifier.
     *
     * @return the ID string
     */
    public String getId() {
        return id;
    }

    /**
     * Gets the identity name.
     *
     * @return the name
     */
    public String getName() {
        return name;
    }

    /**
     * Sets the identity name.
     *
     * @param name the new name
     */
    public void setName(String name) {
        this.name = Objects.requireNonNull(name);
        this.updatedAt = LocalDateTime.now();
    }

    /**
     * Gets the external system ID if set.
     *
     * @return the external ID or null
     */
    public String getExternalId() {
        return externalId;
    }

    /**
     * Sets the external system ID.
     *
     * @param externalId the external ID
     */
    public void setExternalId(String externalId) {
        this.externalId = externalId;
        this.updatedAt = LocalDateTime.now();
    }

    /**
     * Gets the creation timestamp.
     *
     * @return the creation time
     */
    public LocalDateTime getCreatedAt() {
        return createdAt;
    }

    /**
     * Gets the last update timestamp.
     *
     * @return the last update time
     */
    public LocalDateTime getUpdatedAt() {
        return updatedAt;
    }

    /**
     * Checks if this identity is active.
     *
     * @return true if active
     */
    public boolean isActive() {
        return active;
    }

    /**
     * Sets the active status.
     *
     * @param active the active status
     */
    public void setActive(boolean active) {
        this.active = active;
        this.updatedAt = LocalDateTime.now();
    }

    /**
     * Enrolls a new face sample for this identity.
     *
     * @param features the extracted feature vector
     * @param qualityScore the face quality score
     * @param sourceDescription description of the source
     * @return the created EnrolledSample
     */
    public EnrolledSample enrollSample(FeatureVector features, double qualityScore, String sourceDescription) {
        EnrolledSample sample = new EnrolledSample(features, qualityScore, sourceDescription);
        samples.add(sample);
        this.updatedAt = LocalDateTime.now();
        return sample;
    }

    /**
     * Removes a sample by ID.
     *
     * @param sampleId the sample ID to remove
     * @return true if removed
     */
    public boolean removeSample(String sampleId) {
        boolean removed = samples.removeIf(s -> s.getSampleId().equals(sampleId));
        if (removed) {
            this.updatedAt = LocalDateTime.now();
        }
        return removed;
    }

    /**
     * Gets all enrolled samples.
     *
     * @return unmodifiable list of samples
     */
    public List<EnrolledSample> getSamples() {
        return Collections.unmodifiableList(samples);
    }

    /**
     * Gets the number of enrolled samples.
     *
     * @return the sample count
     */
    public int getSampleCount() {
        return samples.size();
    }

    /**
     * Checks if this identity has any enrolled samples.
     *
     * @return true if at least one sample is enrolled
     */
    public boolean hasSamples() {
        return !samples.isEmpty();
    }

    /**
     * Gets all feature vectors for this identity.
     *
     * @return list of feature vectors
     */
    public List<FeatureVector> getAllFeatureVectors() {
        List<FeatureVector> vectors = new ArrayList<>();
        for (EnrolledSample sample : samples) {
            vectors.add(sample.getFeatures());
        }
        return vectors;
    }

    /**
     * Computes the average (centroid) feature vector of all samples.
     *
     * @return the average feature vector, or null if no samples
     */
    public FeatureVector getAverageFeatureVector() {
        if (samples.isEmpty()) {
            return null;
        }

        int dimension = samples.get(0).getFeatures().getDimension();
        double[] sum = new double[dimension];

        for (EnrolledSample sample : samples) {
            double[] features = sample.getFeatures().getFeatures();
            for (int i = 0; i < dimension; i++) {
                sum[i] += features[i];
            }
        }

        for (int i = 0; i < dimension; i++) {
            sum[i] /= samples.size();
        }

        String algorithm = samples.get(0).getFeatures().getAlgorithmName();
        int version = samples.get(0).getFeatures().getAlgorithmVersion();
        return new FeatureVector(sum, algorithm, version);
    }

    /**
     * Gets the best quality sample.
     *
     * @return the sample with highest quality score, or null
     */
    public EnrolledSample getBestQualitySample() {
        return samples.stream()
            .max(Comparator.comparingDouble(EnrolledSample::getQualityScore))
            .orElse(null);
    }

    /**
     * Gets the average quality score across all samples.
     *
     * @return the average quality score
     */
    public double getAverageQualityScore() {
        return samples.stream()
            .mapToDouble(EnrolledSample::getQualityScore)
            .average()
            .orElse(0.0);
    }

    /**
     * Sets a metadata value.
     *
     * @param key the metadata key
     * @param value the metadata value
     */
    public void setMetadata(String key, String value) {
        metadata.put(key, value);
        this.updatedAt = LocalDateTime.now();
    }

    /**
     * Gets a metadata value.
     *
     * @param key the metadata key
     * @return the value, or null if not set
     */
    public String getMetadata(String key) {
        return metadata.get(key);
    }

    /**
     * Gets all metadata.
     *
     * @return unmodifiable metadata map
     */
    public Map<String, String> getAllMetadata() {
        return Collections.unmodifiableMap(metadata);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Identity identity = (Identity) o;
        return id.equals(identity.id);
    }

    @Override
    public int hashCode() {
        return Objects.hash(id);
    }

    @Override
    public String toString() {
        return String.format("Identity{id='%s', name='%s', samples=%d, active=%s}",
            id.substring(0, 8), name, samples.size(), active);
    }
}
