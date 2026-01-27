package com.facerecognition.domain.model;

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.NullAndEmptySource;
import org.junit.jupiter.params.provider.ValueSource;

import java.time.LocalDateTime;
import java.util.List;
import java.util.Map;

import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive unit tests for Identity domain model.
 */
@DisplayName("Identity Tests")
class IdentityTest {

    private static final String TEST_NAME = "John Doe";
    private static final String TEST_EXTERNAL_ID = "EXT-001";

    @Nested
    @DisplayName("Sample Enrollment")
    class SampleEnrollment {

        @Test
        @DisplayName("Should enroll a sample successfully")
        void shouldEnrollSampleSuccessfully() {
            Identity identity = new Identity(TEST_NAME);
            FeatureVector features = createTestFeatureVector(128);

            Identity.EnrolledSample sample = identity.enrollSample(features, 0.95, "test-source");

            assertThat(sample).isNotNull();
            assertThat(sample.getSampleId()).isNotNull().isNotEmpty();
            assertThat(sample.getFeatures()).isEqualTo(features);
            assertThat(sample.getQualityScore()).isEqualTo(0.95);
            assertThat(sample.getSourceDescription()).isEqualTo("test-source");
            assertThat(sample.getEnrolledAt()).isBeforeOrEqualTo(LocalDateTime.now());
        }

        @Test
        @DisplayName("Should track multiple enrolled samples")
        void shouldTrackMultipleEnrolledSamples() {
            Identity identity = new Identity(TEST_NAME);

            identity.enrollSample(createTestFeatureVector(128), 0.9, "source-1");
            identity.enrollSample(createTestFeatureVector(128), 0.85, "source-2");
            identity.enrollSample(createTestFeatureVector(128), 0.95, "source-3");

            assertThat(identity.getSampleCount()).isEqualTo(3);
            assertThat(identity.getSamples()).hasSize(3);
        }

        @Test
        @DisplayName("Should return unmodifiable list of samples")
        void shouldReturnUnmodifiableListOfSamples() {
            Identity identity = new Identity(TEST_NAME);
            identity.enrollSample(createTestFeatureVector(128), 0.9, "source");

            List<Identity.EnrolledSample> samples = identity.getSamples();

            assertThatThrownBy(() -> samples.add(null))
                .isInstanceOf(UnsupportedOperationException.class);
        }

        @Test
        @DisplayName("Should update timestamp on enrollment")
        void shouldUpdateTimestampOnEnrollment() throws InterruptedException {
            Identity identity = new Identity(TEST_NAME);
            LocalDateTime beforeEnroll = identity.getUpdatedAt();

            Thread.sleep(10); // Small delay to ensure different timestamp
            identity.enrollSample(createTestFeatureVector(128), 0.9, "source");

            assertThat(identity.getUpdatedAt()).isAfter(beforeEnroll);
        }

        @Test
        @DisplayName("Should remove sample by ID")
        void shouldRemoveSampleById() {
            Identity identity = new Identity(TEST_NAME);
            Identity.EnrolledSample sample = identity.enrollSample(createTestFeatureVector(128), 0.9, "source");

            boolean removed = identity.removeSample(sample.getSampleId());

            assertThat(removed).isTrue();
            assertThat(identity.getSampleCount()).isZero();
        }

        @Test
        @DisplayName("Should return false when removing non-existent sample")
        void shouldReturnFalseWhenRemovingNonExistentSample() {
            Identity identity = new Identity(TEST_NAME);
            identity.enrollSample(createTestFeatureVector(128), 0.9, "source");

            boolean removed = identity.removeSample("non-existent-id");

            assertThat(removed).isFalse();
            assertThat(identity.getSampleCount()).isEqualTo(1);
        }

        @Test
        @DisplayName("Should handle null source description")
        void shouldHandleNullSourceDescription() {
            Identity identity = new Identity(TEST_NAME);

            Identity.EnrolledSample sample = identity.enrollSample(createTestFeatureVector(128), 0.9, null);

            assertThat(sample.getSourceDescription()).isEmpty();
        }

        @Test
        @DisplayName("Should throw exception for null features")
        void shouldThrowExceptionForNullFeatures() {
            Identity identity = new Identity(TEST_NAME);

            assertThatNullPointerException()
                .isThrownBy(() -> identity.enrollSample(null, 0.9, "source"));
        }

        @Test
        @DisplayName("Should indicate when identity has samples")
        void shouldIndicateWhenIdentityHasSamples() {
            Identity identity = new Identity(TEST_NAME);

            assertThat(identity.hasSamples()).isFalse();

            identity.enrollSample(createTestFeatureVector(128), 0.9, "source");

            assertThat(identity.hasSamples()).isTrue();
        }
    }

    @Nested
    @DisplayName("Feature Vector Averaging")
    class FeatureVectorAveraging {

        @Test
        @DisplayName("Should return null when no samples")
        void shouldReturnNullWhenNoSamples() {
            Identity identity = new Identity(TEST_NAME);

            FeatureVector average = identity.getAverageFeatureVector();

            assertThat(average).isNull();
        }

        @Test
        @DisplayName("Should return same vector for single sample")
        void shouldReturnSameVectorForSingleSample() {
            Identity identity = new Identity(TEST_NAME);
            double[] features = {1.0, 2.0, 3.0, 4.0};
            identity.enrollSample(new FeatureVector(features, "test", 1), 0.9, "source");

            FeatureVector average = identity.getAverageFeatureVector();

            assertThat(average.getFeatures()).containsExactly(1.0, 2.0, 3.0, 4.0);
        }

        @Test
        @DisplayName("Should calculate average correctly for multiple samples")
        void shouldCalculateAverageCorrectlyForMultipleSamples() {
            Identity identity = new Identity(TEST_NAME);
            identity.enrollSample(new FeatureVector(new double[]{1.0, 2.0, 3.0}, "test", 1), 0.9, "s1");
            identity.enrollSample(new FeatureVector(new double[]{3.0, 4.0, 5.0}, "test", 1), 0.9, "s2");

            FeatureVector average = identity.getAverageFeatureVector();

            assertThat(average.getFeatures()).containsExactly(2.0, 3.0, 4.0);
        }

        @Test
        @DisplayName("Should preserve algorithm info in average vector")
        void shouldPreserveAlgorithmInfoInAverageVector() {
            Identity identity = new Identity(TEST_NAME);
            identity.enrollSample(new FeatureVector(new double[]{1.0, 2.0}, "Eigenfaces", 2), 0.9, "s1");

            FeatureVector average = identity.getAverageFeatureVector();

            assertThat(average.getAlgorithmName()).isEqualTo("Eigenfaces");
            assertThat(average.getAlgorithmVersion()).isEqualTo(2);
        }

        @Test
        @DisplayName("Should get all feature vectors")
        void shouldGetAllFeatureVectors() {
            Identity identity = new Identity(TEST_NAME);
            FeatureVector fv1 = new FeatureVector(new double[]{1.0, 2.0}, "test", 1);
            FeatureVector fv2 = new FeatureVector(new double[]{3.0, 4.0}, "test", 1);
            identity.enrollSample(fv1, 0.9, "s1");
            identity.enrollSample(fv2, 0.9, "s2");

            List<FeatureVector> vectors = identity.getAllFeatureVectors();

            assertThat(vectors).hasSize(2);
            assertThat(vectors).containsExactly(fv1, fv2);
        }
    }

    @Nested
    @DisplayName("Quality Score Management")
    class QualityScoreManagement {

        @Test
        @DisplayName("Should get best quality sample")
        void shouldGetBestQualitySample() {
            Identity identity = new Identity(TEST_NAME);
            identity.enrollSample(createTestFeatureVector(10), 0.7, "low");
            identity.enrollSample(createTestFeatureVector(10), 0.95, "high");
            identity.enrollSample(createTestFeatureVector(10), 0.85, "medium");

            Identity.EnrolledSample best = identity.getBestQualitySample();

            assertThat(best).isNotNull();
            assertThat(best.getQualityScore()).isEqualTo(0.95);
            assertThat(best.getSourceDescription()).isEqualTo("high");
        }

        @Test
        @DisplayName("Should return null for best quality when no samples")
        void shouldReturnNullForBestQualityWhenNoSamples() {
            Identity identity = new Identity(TEST_NAME);

            Identity.EnrolledSample best = identity.getBestQualitySample();

            assertThat(best).isNull();
        }

        @Test
        @DisplayName("Should calculate average quality score")
        void shouldCalculateAverageQualityScore() {
            Identity identity = new Identity(TEST_NAME);
            identity.enrollSample(createTestFeatureVector(10), 0.8, "s1");
            identity.enrollSample(createTestFeatureVector(10), 0.9, "s2");
            identity.enrollSample(createTestFeatureVector(10), 1.0, "s3");

            double avgQuality = identity.getAverageQualityScore();

            assertThat(avgQuality).isCloseTo(0.9, within(0.001));
        }

        @Test
        @DisplayName("Should return 0.0 for average quality when no samples")
        void shouldReturnZeroForAverageQualityWhenNoSamples() {
            Identity identity = new Identity(TEST_NAME);

            double avgQuality = identity.getAverageQualityScore();

            assertThat(avgQuality).isEqualTo(0.0);
        }
    }

    @Nested
    @DisplayName("Metadata Handling")
    class MetadataHandling {

        @Test
        @DisplayName("Should set and get metadata")
        void shouldSetAndGetMetadata() {
            Identity identity = new Identity(TEST_NAME);

            identity.setMetadata("department", "Engineering");
            identity.setMetadata("role", "Developer");

            assertThat(identity.getMetadata("department")).isEqualTo("Engineering");
            assertThat(identity.getMetadata("role")).isEqualTo("Developer");
        }

        @Test
        @DisplayName("Should return null for non-existent metadata key")
        void shouldReturnNullForNonExistentMetadataKey() {
            Identity identity = new Identity(TEST_NAME);

            assertThat(identity.getMetadata("non-existent")).isNull();
        }

        @Test
        @DisplayName("Should overwrite existing metadata")
        void shouldOverwriteExistingMetadata() {
            Identity identity = new Identity(TEST_NAME);
            identity.setMetadata("key", "value1");

            identity.setMetadata("key", "value2");

            assertThat(identity.getMetadata("key")).isEqualTo("value2");
        }

        @Test
        @DisplayName("Should get all metadata as unmodifiable map")
        void shouldGetAllMetadataAsUnmodifiableMap() {
            Identity identity = new Identity(TEST_NAME);
            identity.setMetadata("key1", "value1");
            identity.setMetadata("key2", "value2");

            Map<String, String> metadata = identity.getAllMetadata();

            assertThat(metadata).hasSize(2);
            assertThat(metadata).containsEntry("key1", "value1");
            assertThat(metadata).containsEntry("key2", "value2");
            assertThatThrownBy(() -> metadata.put("key3", "value3"))
                .isInstanceOf(UnsupportedOperationException.class);
        }

        @Test
        @DisplayName("Should update timestamp on metadata change")
        void shouldUpdateTimestampOnMetadataChange() throws InterruptedException {
            Identity identity = new Identity(TEST_NAME);
            LocalDateTime before = identity.getUpdatedAt();

            Thread.sleep(10);
            identity.setMetadata("key", "value");

            assertThat(identity.getUpdatedAt()).isAfter(before);
        }
    }

    @Nested
    @DisplayName("Identity Construction")
    class IdentityConstruction {

        @Test
        @DisplayName("Should create identity with name only")
        void shouldCreateIdentityWithNameOnly() {
            Identity identity = new Identity(TEST_NAME);

            assertThat(identity.getId()).isNotNull().isNotEmpty();
            assertThat(identity.getName()).isEqualTo(TEST_NAME);
            assertThat(identity.getExternalId()).isNull();
            assertThat(identity.isActive()).isTrue();
            assertThat(identity.getCreatedAt()).isBeforeOrEqualTo(LocalDateTime.now());
        }

        @Test
        @DisplayName("Should create identity with ID, name, and external ID")
        void shouldCreateIdentityWithAllParams() {
            String customId = "custom-id-123";
            Identity identity = new Identity(customId, TEST_NAME, TEST_EXTERNAL_ID);

            assertThat(identity.getId()).isEqualTo(customId);
            assertThat(identity.getName()).isEqualTo(TEST_NAME);
            assertThat(identity.getExternalId()).isEqualTo(TEST_EXTERNAL_ID);
        }

        @Test
        @DisplayName("Should throw exception for null ID")
        void shouldThrowExceptionForNullId() {
            assertThatNullPointerException()
                .isThrownBy(() -> new Identity(null, TEST_NAME, null))
                .withMessage("ID cannot be null");
        }

        @Test
        @DisplayName("Should throw exception for null name")
        void shouldThrowExceptionForNullName() {
            assertThatNullPointerException()
                .isThrownBy(() -> new Identity(null))
                .withMessage("Name cannot be null");
        }

        @Test
        @DisplayName("Should allow null external ID")
        void shouldAllowNullExternalId() {
            Identity identity = new Identity("id", TEST_NAME, null);

            assertThat(identity.getExternalId()).isNull();
        }
    }

    @Nested
    @DisplayName("Identity State Management")
    class IdentityStateManagement {

        @Test
        @DisplayName("Should set name and update timestamp")
        void shouldSetNameAndUpdateTimestamp() throws InterruptedException {
            Identity identity = new Identity(TEST_NAME);
            LocalDateTime before = identity.getUpdatedAt();

            Thread.sleep(10);
            identity.setName("Jane Doe");

            assertThat(identity.getName()).isEqualTo("Jane Doe");
            assertThat(identity.getUpdatedAt()).isAfter(before);
        }

        @Test
        @DisplayName("Should throw exception for null name on set")
        void shouldThrowExceptionForNullNameOnSet() {
            Identity identity = new Identity(TEST_NAME);

            assertThatNullPointerException()
                .isThrownBy(() -> identity.setName(null));
        }

        @Test
        @DisplayName("Should set external ID")
        void shouldSetExternalId() {
            Identity identity = new Identity(TEST_NAME);

            identity.setExternalId(TEST_EXTERNAL_ID);

            assertThat(identity.getExternalId()).isEqualTo(TEST_EXTERNAL_ID);
        }

        @Test
        @DisplayName("Should toggle active status")
        void shouldToggleActiveStatus() {
            Identity identity = new Identity(TEST_NAME);

            assertThat(identity.isActive()).isTrue();

            identity.setActive(false);
            assertThat(identity.isActive()).isFalse();

            identity.setActive(true);
            assertThat(identity.isActive()).isTrue();
        }
    }

    @Nested
    @DisplayName("Equality and HashCode")
    class EqualityAndHashCode {

        @Test
        @DisplayName("Should be equal when same ID")
        void shouldBeEqualWhenSameId() {
            Identity identity1 = new Identity("same-id", "Name1", null);
            Identity identity2 = new Identity("same-id", "Name2", null);

            assertThat(identity1).isEqualTo(identity2);
        }

        @Test
        @DisplayName("Should not be equal when different ID")
        void shouldNotBeEqualWhenDifferentId() {
            Identity identity1 = new Identity(TEST_NAME);
            Identity identity2 = new Identity(TEST_NAME);

            assertThat(identity1).isNotEqualTo(identity2);
        }

        @Test
        @DisplayName("Should have same hashCode when same ID")
        void shouldHaveSameHashCodeWhenSameId() {
            Identity identity1 = new Identity("same-id", "Name1", null);
            Identity identity2 = new Identity("same-id", "Name2", null);

            assertThat(identity1.hashCode()).isEqualTo(identity2.hashCode());
        }

        @Test
        @DisplayName("Should not be equal to null")
        void shouldNotBeEqualToNull() {
            Identity identity = new Identity(TEST_NAME);

            assertThat(identity).isNotEqualTo(null);
        }
    }

    @Nested
    @DisplayName("ToString")
    class ToStringTests {

        @Test
        @DisplayName("Should include relevant information")
        void shouldIncludeRelevantInformation() {
            Identity identity = new Identity(TEST_NAME);
            identity.enrollSample(createTestFeatureVector(10), 0.9, "source");

            String str = identity.toString();

            assertThat(str).contains("Identity");
            assertThat(str).contains(TEST_NAME);
            assertThat(str).contains("samples=1");
            assertThat(str).contains("active=true");
        }
    }

    @Nested
    @DisplayName("EnrolledSample ToString")
    class EnrolledSampleToStringTests {

        @Test
        @DisplayName("Should format enrolled sample toString correctly")
        void shouldFormatEnrolledSampleToStringCorrectly() {
            Identity identity = new Identity(TEST_NAME);
            Identity.EnrolledSample sample = identity.enrollSample(createTestFeatureVector(10), 0.95, "test");

            String str = sample.toString();

            assertThat(str).contains("EnrolledSample");
            assertThat(str).contains("quality=0.95");
        }
    }

    // Helper methods

    private FeatureVector createTestFeatureVector(int dimension) {
        double[] features = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            features[i] = Math.random();
        }
        return new FeatureVector(features, "test", 1);
    }
}
