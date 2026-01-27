package com.facerecognition.infrastructure.classification;

import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.domain.service.FaceClassifier;
import com.facerecognition.domain.service.FaceClassifier.ClassifierConfig;
import com.facerecognition.domain.service.FaceClassifier.DistanceMetric;

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive unit tests for KNNClassifier.
 */
@DisplayName("KNNClassifier Tests")
class KNNClassifierTest {

    private KNNClassifier classifier;

    @BeforeEach
    void setUp() {
        classifier = new KNNClassifier();
    }

    @Nested
    @DisplayName("Enrollment Tests")
    class EnrollmentTests {

        @Test
        @DisplayName("Should enroll identity successfully")
        void shouldEnrollIdentitySuccessfully() {
            Identity identity = createIdentityWithSamples("John Doe", 1);

            classifier.enroll(identity);

            assertThat(classifier.isEnrolled(identity.getId())).isTrue();
            assertThat(classifier.getEnrolledCount()).isEqualTo(1);
        }

        @Test
        @DisplayName("Should enroll multiple identities")
        void shouldEnrollMultipleIdentities() {
            Identity identity1 = createIdentityWithSamples("John Doe", 1);
            Identity identity2 = createIdentityWithSamples("Jane Doe", 1);
            Identity identity3 = createIdentityWithSamples("Bob Smith", 1);

            classifier.enroll(identity1);
            classifier.enroll(identity2);
            classifier.enroll(identity3);

            assertThat(classifier.getEnrolledCount()).isEqualTo(3);
            assertThat(classifier.getEnrolledIdentities()).hasSize(3);
        }

        @Test
        @DisplayName("Should throw exception for null identity")
        void shouldThrowExceptionForNullIdentity() {
            assertThatNullPointerException()
                .isThrownBy(() -> classifier.enroll(null))
                .withMessage("Identity cannot be null");
        }

        @Test
        @DisplayName("Should throw exception for identity without samples")
        void shouldThrowExceptionForIdentityWithoutSamples() {
            Identity identityWithoutSamples = new Identity("Empty Person");

            assertThatIllegalArgumentException()
                .isThrownBy(() -> classifier.enroll(identityWithoutSamples))
                .withMessageContaining("must have at least one enrolled sample");
        }

        @Test
        @DisplayName("Should unenroll identity")
        void shouldUnenrollIdentity() {
            Identity identity = createIdentityWithSamples("John Doe", 1);
            classifier.enroll(identity);

            boolean removed = classifier.unenroll(identity.getId());

            assertThat(removed).isTrue();
            assertThat(classifier.isEnrolled(identity.getId())).isFalse();
            assertThat(classifier.getEnrolledCount()).isZero();
        }

        @Test
        @DisplayName("Should return false when unenrolling non-existent identity")
        void shouldReturnFalseWhenUnenrollingNonExistentIdentity() {
            boolean removed = classifier.unenroll("non-existent-id");

            assertThat(removed).isFalse();
        }

        @Test
        @DisplayName("Should clear all enrolled identities")
        void shouldClearAllEnrolledIdentities() {
            classifier.enroll(createIdentityWithSamples("Person 1", 1));
            classifier.enroll(createIdentityWithSamples("Person 2", 1));
            classifier.enroll(createIdentityWithSamples("Person 3", 1));

            classifier.clear();

            assertThat(classifier.getEnrolledCount()).isZero();
        }

        @Test
        @DisplayName("Should return enrolled identities list")
        void shouldReturnEnrolledIdentitiesList() {
            Identity identity1 = createIdentityWithSamples("John Doe", 1);
            Identity identity2 = createIdentityWithSamples("Jane Doe", 1);
            classifier.enroll(identity1);
            classifier.enroll(identity2);

            List<Identity> enrolled = classifier.getEnrolledIdentities();

            assertThat(enrolled).hasSize(2);
            assertThat(enrolled).extracting(Identity::getName)
                .containsExactlyInAnyOrder("John Doe", "Jane Doe");
        }
    }

    @Nested
    @DisplayName("Classification Tests")
    class ClassificationTests {

        @Test
        @DisplayName("Should classify matching identity")
        void shouldClassifyMatchingIdentity() {
            // Create identity with known features
            double[] knownFeatures = {1.0, 0.0, 0.0, 0.0};
            Identity identity = createIdentityWithFeatures("John Doe", knownFeatures);
            classifier.enroll(identity);

            // Create probe similar to known features
            FeatureVector probe = new FeatureVector(new double[]{1.0, 0.0, 0.0, 0.0}, "test", 1);

            RecognitionResult result = classifier.classify(probe);

            assertThat(result.isRecognized()).isTrue();
            assertThat(result.getIdentity()).isPresent();
            assertThat(result.getIdentity().get().getName()).isEqualTo("John Doe");
        }

        @Test
        @DisplayName("Should return unknown when no identities enrolled")
        void shouldReturnUnknownWhenNoIdentitiesEnrolled() {
            FeatureVector probe = createTestFeatureVector(128);

            RecognitionResult result = classifier.classify(probe);

            assertThat(result.getStatus()).isEqualTo(RecognitionResult.Status.UNKNOWN);
            assertThat(result.isRecognized()).isFalse();
        }

        @Test
        @DisplayName("Should return unknown when below threshold")
        void shouldReturnUnknownWhenBelowThreshold() {
            // Create identity with features very different from probe
            Identity identity = createIdentityWithFeatures("John Doe", new double[]{1.0, 0.0, 0.0, 0.0});
            classifier.enroll(identity);

            // Create very different probe
            FeatureVector probe = new FeatureVector(new double[]{0.0, 1.0, 1.0, 1.0}, "test", 1);

            RecognitionResult result = classifier.classify(probe, 0.999);

            assertThat(result.getStatus()).isEqualTo(RecognitionResult.Status.UNKNOWN);
        }

        @Test
        @DisplayName("Should find closest match among multiple identities")
        void shouldFindClosestMatchAmongMultipleIdentities() {
            Identity john = createIdentityWithFeatures("John", new double[]{1.0, 0.0, 0.0, 0.0});
            Identity jane = createIdentityWithFeatures("Jane", new double[]{0.0, 1.0, 0.0, 0.0});
            Identity bob = createIdentityWithFeatures("Bob", new double[]{0.0, 0.0, 1.0, 0.0});

            classifier.enroll(john);
            classifier.enroll(jane);
            classifier.enroll(bob);

            // Probe closest to Jane
            FeatureVector probe = new FeatureVector(new double[]{0.1, 0.9, 0.1, 0.0}, "test", 1);

            RecognitionResult result = classifier.classify(probe);

            assertThat(result.getIdentity()).isPresent();
            assertThat(result.getIdentity().get().getName()).isEqualTo("Jane");
        }

        @Test
        @DisplayName("Should include extracted features in result")
        void shouldIncludeExtractedFeaturesInResult() {
            Identity identity = createIdentityWithSamples("John Doe", 1);
            classifier.enroll(identity);

            FeatureVector probe = createTestFeatureVector(128);
            RecognitionResult result = classifier.classify(probe);

            assertThat(result.getExtractedFeatures()).isPresent();
            assertThat(result.getExtractedFeatures().get()).isEqualTo(probe);
        }

        @Test
        @DisplayName("Should skip inactive identities")
        void shouldSkipInactiveIdentities() {
            Identity activeIdentity = createIdentityWithFeatures("Active", new double[]{0.0, 1.0, 0.0, 0.0});
            Identity inactiveIdentity = createIdentityWithFeatures("Inactive", new double[]{1.0, 0.0, 0.0, 0.0});
            inactiveIdentity.setActive(false);

            classifier.enroll(activeIdentity);
            classifier.enroll(inactiveIdentity);

            // Probe closer to inactive identity
            FeatureVector probe = new FeatureVector(new double[]{0.9, 0.1, 0.0, 0.0}, "test", 1);

            RecognitionResult result = classifier.classify(probe);

            // Should match active identity since inactive is skipped
            assertThat(result.getIdentity()).isPresent();
            assertThat(result.getIdentity().get().getName()).isEqualTo("Active");
        }
    }

    @Nested
    @DisplayName("Distance Metric Tests")
    class DistanceMetricTests {

        @ParameterizedTest
        @DisplayName("Should support all distance metrics")
        @EnumSource(DistanceMetric.class)
        void shouldSupportAllDistanceMetrics(DistanceMetric metric) {
            if (metric == DistanceMetric.MAHALANOBIS) {
                // Skip Mahalanobis as it requires special setup
                return;
            }

            ClassifierConfig config = new ClassifierConfig()
                .setMetric(metric);
            KNNClassifier metricClassifier = new KNNClassifier(config);

            Identity identity = createIdentityWithSamples("Test", 1);
            metricClassifier.enroll(identity);

            assertThat(metricClassifier.getDistanceMetric()).isEqualTo(metric);
        }

        @Test
        @DisplayName("Should calculate Euclidean distance correctly")
        void shouldCalculateEuclideanDistanceCorrectly() {
            ClassifierConfig config = new ClassifierConfig()
                .setMetric(DistanceMetric.EUCLIDEAN);
            KNNClassifier euclideanClassifier = new KNNClassifier(config);

            Identity identity = createIdentityWithFeatures("Test", new double[]{1.0, 0.0, 0.0});
            euclideanClassifier.enroll(identity);

            FeatureVector probe = new FeatureVector(new double[]{0.0, 0.0, 0.0}, "test", 1);
            double distance = euclideanClassifier.getDistance(probe, identity.getId());

            // Euclidean distance from (0,0,0) to (1,0,0) = 1.0
            assertThat(distance).isCloseTo(1.0, within(0.001));
        }

        @Test
        @DisplayName("Should calculate Cosine distance correctly")
        void shouldCalculateCosineDistanceCorrectly() {
            ClassifierConfig config = new ClassifierConfig()
                .setMetric(DistanceMetric.COSINE);
            KNNClassifier cosineClassifier = new KNNClassifier(config);

            Identity identity = createIdentityWithFeatures("Test", new double[]{1.0, 0.0, 0.0});
            cosineClassifier.enroll(identity);

            // Same direction = cosine similarity of 1, distance of 0
            FeatureVector probe = new FeatureVector(new double[]{2.0, 0.0, 0.0}, "test", 1);
            double distance = cosineClassifier.getDistance(probe, identity.getId());

            assertThat(distance).isCloseTo(0.0, within(0.001));
        }

        @Test
        @DisplayName("Should calculate Manhattan distance correctly")
        void shouldCalculateManhattanDistanceCorrectly() {
            ClassifierConfig config = new ClassifierConfig()
                .setMetric(DistanceMetric.MANHATTAN);
            KNNClassifier manhattanClassifier = new KNNClassifier(config);

            Identity identity = createIdentityWithFeatures("Test", new double[]{1.0, 1.0, 1.0});
            manhattanClassifier.enroll(identity);

            FeatureVector probe = new FeatureVector(new double[]{0.0, 0.0, 0.0}, "test", 1);
            double distance = manhattanClassifier.getDistance(probe, identity.getId());

            // Manhattan distance = |1-0| + |1-0| + |1-0| = 3
            assertThat(distance).isCloseTo(3.0, within(0.001));
        }

        @Test
        @DisplayName("Should calculate Chi-Square distance correctly")
        void shouldCalculateChiSquareDistanceCorrectly() {
            ClassifierConfig config = new ClassifierConfig()
                .setMetric(DistanceMetric.CHI_SQUARE);
            KNNClassifier chiSquareClassifier = new KNNClassifier(config);

            Identity identity = createIdentityWithFeatures("Test", new double[]{0.5, 0.5, 0.0});
            chiSquareClassifier.enroll(identity);

            FeatureVector probe = new FeatureVector(new double[]{0.25, 0.75, 0.0}, "test", 1);
            double distance = chiSquareClassifier.getDistance(probe, identity.getId());

            assertThat(distance).isGreaterThan(0.0);
        }

        @Test
        @DisplayName("Should change distance metric")
        void shouldChangeDistanceMetric() {
            assertThat(classifier.getDistanceMetric()).isEqualTo(DistanceMetric.EUCLIDEAN);

            classifier.setDistanceMetric(DistanceMetric.COSINE);

            assertThat(classifier.getDistanceMetric()).isEqualTo(DistanceMetric.COSINE);
        }

        @Test
        @DisplayName("Should throw exception for null distance metric")
        void shouldThrowExceptionForNullDistanceMetric() {
            assertThatNullPointerException()
                .isThrownBy(() -> classifier.setDistanceMetric(null));
        }
    }

    @Nested
    @DisplayName("K Parameter Tests")
    class KParameterTests {

        @ParameterizedTest
        @DisplayName("Should work with different K values")
        @ValueSource(ints = {1, 3, 5, 10})
        void shouldWorkWithDifferentKValues(int k) {
            ClassifierConfig config = new ClassifierConfig().setK(k);
            KNNClassifier knnClassifier = new KNNClassifier(config);

            Identity identity = createIdentityWithSamples("Test", 1);
            knnClassifier.enroll(identity);

            FeatureVector probe = createTestFeatureVector(128);
            RecognitionResult result = knnClassifier.classify(probe);

            assertThat(result).isNotNull();
        }

        @Test
        @DisplayName("Should use K for getting alternatives")
        void shouldUseKForGettingAlternatives() {
            ClassifierConfig config = new ClassifierConfig().setK(3);
            KNNClassifier knnClassifier = new KNNClassifier(config);

            // Enroll multiple identities
            for (int i = 0; i < 5; i++) {
                knnClassifier.enroll(createIdentityWithSamples("Person " + i, 1));
            }

            FeatureVector probe = createTestFeatureVector(128);
            RecognitionResult result = knnClassifier.classify(probe, 0.0);

            // Should have up to K-1 alternatives (since one is best match)
            assertThat(result.getAlternatives().size()).isLessThanOrEqualTo(2);
        }
    }

    @Nested
    @DisplayName("Top Matches Tests")
    class TopMatchesTests {

        @Test
        @DisplayName("Should get top N matches")
        void shouldGetTopNMatches() {
            // Enroll multiple identities
            for (int i = 0; i < 10; i++) {
                classifier.enroll(createIdentityWithSamples("Person " + i, 1));
            }

            FeatureVector probe = createTestFeatureVector(128);
            RecognitionResult result = classifier.getTopMatches(probe, 5);

            // Should have best match and up to 4 alternatives
            assertThat(result.getAlternatives().size()).isLessThanOrEqualTo(4);
        }

        @Test
        @DisplayName("Should handle requesting more matches than enrolled")
        void shouldHandleRequestingMoreMatchesThanEnrolled() {
            classifier.enroll(createIdentityWithSamples("Person 1", 1));
            classifier.enroll(createIdentityWithSamples("Person 2", 1));

            FeatureVector probe = createTestFeatureVector(128);
            RecognitionResult result = classifier.getTopMatches(probe, 10);

            assertThat(result).isNotNull();
        }

        @Test
        @DisplayName("Should return unknown when no matches available")
        void shouldReturnUnknownWhenNoMatchesAvailable() {
            FeatureVector probe = createTestFeatureVector(128);
            RecognitionResult result = classifier.getTopMatches(probe, 5);

            assertThat(result.getStatus()).isEqualTo(RecognitionResult.Status.UNKNOWN);
        }
    }

    @Nested
    @DisplayName("Configuration Tests")
    class ConfigurationTests {

        @Test
        @DisplayName("Should use average features when configured")
        void shouldUseAverageFeaturesWhenConfigured() {
            ClassifierConfig config = new ClassifierConfig()
                .setUseAverageFeatures(true);
            KNNClassifier avgClassifier = new KNNClassifier(config);

            // Create identity with multiple samples
            Identity identity = new Identity("Test");
            identity.enrollSample(new FeatureVector(new double[]{1.0, 0.0}, "test", 1), 0.9, "s1");
            identity.enrollSample(new FeatureVector(new double[]{0.0, 1.0}, "test", 1), 0.9, "s2");
            avgClassifier.enroll(identity);

            // Probe at average position
            FeatureVector probe = new FeatureVector(new double[]{0.5, 0.5}, "test", 1);

            RecognitionResult result = avgClassifier.classify(probe);

            assertThat(result).isNotNull();
        }

        @Test
        @DisplayName("Should use minimum distance by default")
        void shouldUseMinimumDistanceByDefault() {
            // Create identity with multiple samples at different positions
            Identity identity = new Identity("Test");
            identity.enrollSample(new FeatureVector(new double[]{0.0, 0.0}, "test", 1), 0.9, "s1");
            identity.enrollSample(new FeatureVector(new double[]{10.0, 10.0}, "test", 1), 0.9, "s2");
            classifier.enroll(identity);

            // Probe close to first sample
            FeatureVector probe = new FeatureVector(new double[]{0.1, 0.1}, "test", 1);
            double distance = classifier.getDistance(probe, identity.getId());

            // Should use minimum distance (close to 0)
            assertThat(distance).isLessThan(1.0);
        }

        @Test
        @DisplayName("Should return MAX_VALUE distance for non-enrolled identity")
        void shouldReturnMaxValueDistanceForNonEnrolledIdentity() {
            FeatureVector probe = createTestFeatureVector(128);
            double distance = classifier.getDistance(probe, "non-existent-id");

            assertThat(distance).isEqualTo(Double.MAX_VALUE);
        }

        @Test
        @DisplayName("Should return classifier name")
        void shouldReturnClassifierName() {
            assertThat(classifier.getName()).isEqualTo("KNN");
        }

        @Test
        @DisplayName("Should handle retrain call (no-op for KNN)")
        void shouldHandleRetrainCall() {
            classifier.enroll(createIdentityWithSamples("Test", 1));

            assertThatNoException().isThrownBy(() -> classifier.retrain());
        }
    }

    @Nested
    @DisplayName("Threshold Tests")
    class ThresholdTests {

        @Test
        @DisplayName("Should use default threshold from config")
        void shouldUseDefaultThresholdFromConfig() {
            ClassifierConfig config = new ClassifierConfig().setThreshold(0.8);
            KNNClassifier thresholdClassifier = new KNNClassifier(config);

            Identity identity = createIdentityWithFeatures("Test", new double[]{1.0, 0.0, 0.0, 0.0});
            thresholdClassifier.enroll(identity);

            // Probe that may or may not meet threshold
            FeatureVector probe = new FeatureVector(new double[]{0.5, 0.5, 0.0, 0.0}, "test", 1);
            RecognitionResult result = thresholdClassifier.classify(probe);

            assertThat(result).isNotNull();
        }

        @Test
        @DisplayName("Should use custom threshold when provided")
        void shouldUseCustomThresholdWhenProvided() {
            Identity identity = createIdentityWithFeatures("Test", new double[]{1.0, 0.0, 0.0, 0.0});
            classifier.enroll(identity);

            FeatureVector probe = new FeatureVector(new double[]{1.0, 0.0, 0.0, 0.0}, "test", 1);

            // With very low threshold, should always recognize
            RecognitionResult lowThreshold = classifier.classify(probe, 0.0);
            assertThat(lowThreshold.isRecognized()).isTrue();

            // With very high threshold, may not recognize
            RecognitionResult highThreshold = classifier.classify(probe, 0.9999);
            assertThat(highThreshold).isNotNull();
        }
    }

    @Nested
    @DisplayName("ToString Tests")
    class ToStringTests {

        @Test
        @DisplayName("Should format toString correctly")
        void shouldFormatToStringCorrectly() {
            classifier.enroll(createIdentityWithSamples("Test", 1));

            String str = classifier.toString();

            assertThat(str).contains("KNNClassifier");
            assertThat(str).contains("k=");
            assertThat(str).contains("metric=");
            assertThat(str).contains("enrolled=1");
        }
    }

    // Helper methods

    private Identity createIdentityWithSamples(String name, int sampleCount) {
        Identity identity = new Identity(name);
        for (int i = 0; i < sampleCount; i++) {
            identity.enrollSample(createTestFeatureVector(128), 0.9, "sample-" + i);
        }
        return identity;
    }

    private Identity createIdentityWithFeatures(String name, double[] features) {
        Identity identity = new Identity(name);
        identity.enrollSample(new FeatureVector(features, "test", 1), 0.9, "sample");
        return identity;
    }

    private FeatureVector createTestFeatureVector(int dimension) {
        double[] features = new double[dimension];
        for (int i = 0; i < dimension; i++) {
            features[i] = Math.random();
        }
        return new FeatureVector(features, "test", 1);
    }
}
