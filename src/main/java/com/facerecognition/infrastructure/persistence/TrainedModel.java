package com.facerecognition.infrastructure.persistence;

import com.facerecognition.domain.service.FeatureExtractor;

import java.io.Serializable;
import java.time.Instant;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Everything needed to restore a recognition service: the fitted extractor, the enrolled
 * identities and the preprocessed grayscale face crops they were trained on.
 *
 * <p>The crops are kept so a restored service can be retrained after new enrollments
 * without losing the identities that were loaded from disk.</p>
 *
 * @param extractor  the feature extractor, trained or not
 * @param identities the enrolled identities
 * @param samples    the training crops, each tagged with its identity id
 * @param createdAt  when this snapshot was taken
 */
public record TrainedModel(
        FeatureExtractor extractor,
        List<IdentityRecord> identities,
        List<FaceSample> samples,
        Instant createdAt) implements Serializable {

    public TrainedModel {
        Objects.requireNonNull(extractor, "extractor");
        Objects.requireNonNull(createdAt, "createdAt");
        identities = List.copyOf(identities);
        samples = List.copyOf(samples);
    }

    /**
     * The persisted part of an {@link com.facerecognition.domain.model.Identity}.
     *
     * @param id         stable identity id
     * @param name       display name
     * @param externalId caller-supplied id, may be null
     * @param active     false once soft-deleted
     * @param metadata   free-form key/value pairs
     */
    public record IdentityRecord(
            String id,
            String name,
            String externalId,
            boolean active,
            Map<String, String> metadata) implements Serializable {

        public IdentityRecord {
            Objects.requireNonNull(id, "id");
            Objects.requireNonNull(name, "name");
            metadata = Collections.unmodifiableMap(new HashMap<>(metadata));
        }
    }

    /**
     * A preprocessed face crop stored as one grayscale byte per pixel, row-major.
     *
     * @param identityId the identity this crop belongs to
     * @param width      crop width in pixels
     * @param height     crop height in pixels
     * @param pixels     grayscale values, length {@code width * height}
     */
    public record FaceSample(String identityId, int width, int height, byte[] pixels)
            implements Serializable {

        public FaceSample {
            Objects.requireNonNull(identityId, "identityId");
            if (width <= 0 || height <= 0 || pixels.length != width * height) {
                throw new IllegalArgumentException(
                        "pixels must hold width * height values: " + width + "x" + height
                                + " != " + pixels.length);
            }
        }
    }
}
