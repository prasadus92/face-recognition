package com.facerecognition.domain.model;

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.time.LocalDateTime;
import java.util.Arrays;
import java.util.List;
import java.util.Optional;

import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive unit tests for RecognitionResult domain model.
 */
@DisplayName("RecognitionResult Tests")
class RecognitionResultTest {

    @Nested
    @DisplayName("Builder Pattern Tests")
    class BuilderPatternTests {

        @Test
        @DisplayName("Should create result with builder")
        void shouldCreateResultWithBuilder() {
            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .build();

            assertThat(result).isNotNull();
            assertThat(result.getStatus()).isEqualTo(RecognitionResult.Status.RECOGNIZED);
            assertThat(result.getRequestId()).isNotNull().isNotEmpty();
            assertThat(result.getTimestamp()).isBeforeOrEqualTo(LocalDateTime.now());
        }

        @Test
        @DisplayName("Should set best match through builder")
        void shouldSetBestMatchThroughBuilder() {
            Identity identity = new Identity("Test Person");
            identity.enrollSample(createTestFeatureVector(), 0.9, "source");
            RecognitionResult.MatchResult match = new RecognitionResult.MatchResult(identity, 0.95, 0.05);

            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .bestMatch(match)
                .build();

            assertThat(result.getBestMatch()).isPresent();
            assertThat(result.getBestMatch().get().getIdentity()).isEqualTo(identity);
            assertThat(result.getBestMatch().get().getConfidence()).isEqualTo(0.95);
            assertThat(result.getBestMatch().get().getDistance()).isEqualTo(0.05);
        }

        @Test
        @DisplayName("Should add alternatives through builder")
        void shouldAddAlternativesThroughBuilder() {
            Identity identity1 = createIdentity("Person1");
            Identity identity2 = createIdentity("Person2");

            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .addAlternative(new RecognitionResult.MatchResult(identity1, 0.8, 0.1))
                .addAlternative(new RecognitionResult.MatchResult(identity2, 0.7, 0.2))
                .build();

            assertThat(result.getAlternatives()).hasSize(2);
        }

        @Test
        @DisplayName("Should set alternatives list through builder")
        void shouldSetAlternativesListThroughBuilder() {
            Identity identity1 = createIdentity("Person1");
            Identity identity2 = createIdentity("Person2");
            List<RecognitionResult.MatchResult> alternatives = Arrays.asList(
                new RecognitionResult.MatchResult(identity1, 0.8, 0.1),
                new RecognitionResult.MatchResult(identity2, 0.7, 0.2)
            );

            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.UNKNOWN)
                .alternatives(alternatives)
                .build();

            assertThat(result.getAlternatives()).hasSize(2);
        }

        @Test
        @DisplayName("Should set detected face through builder")
        void shouldSetDetectedFaceThroughBuilder() {
            FaceRegion faceRegion = new FaceRegion(10, 20, 100, 150, 0.95);

            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .detectedFace(faceRegion)
                .build();

            assertThat(result.getDetectedFace()).isPresent();
            assertThat(result.getDetectedFace().get()).isEqualTo(faceRegion);
        }

        @Test
        @DisplayName("Should set extracted features through builder")
        void shouldSetExtractedFeaturesThroughBuilder() {
            FeatureVector features = createTestFeatureVector();

            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .extractedFeatures(features)
                .build();

            assertThat(result.getExtractedFeatures()).isPresent();
            assertThat(result.getExtractedFeatures().get()).isEqualTo(features);
        }

        @Test
        @DisplayName("Should set metrics through builder")
        void shouldSetMetricsThroughBuilder() {
            RecognitionResult.ProcessingMetrics metrics =
                new RecognitionResult.ProcessingMetrics(10, 20, 30, 60);

            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .metrics(metrics)
                .build();

            assertThat(result.getMetrics()).isPresent();
            assertThat(result.getMetrics().get().getDetectionTimeMs()).isEqualTo(10);
            assertThat(result.getMetrics().get().getExtractionTimeMs()).isEqualTo(20);
            assertThat(result.getMetrics().get().getMatchingTimeMs()).isEqualTo(30);
            assertThat(result.getMetrics().get().getTotalTimeMs()).isEqualTo(60);
        }

        @Test
        @DisplayName("Should set error through builder")
        void shouldSetErrorThroughBuilder() {
            RecognitionResult result = RecognitionResult.builder()
                .error("Recognition failed due to system error")
                .build();

            assertThat(result.getStatus()).isEqualTo(RecognitionResult.Status.ERROR);
            assertThat(result.isError()).isTrue();
            assertThat(result.getErrorMessage()).isPresent();
            assertThat(result.getErrorMessage().get()).isEqualTo("Recognition failed due to system error");
        }

        @Test
        @DisplayName("Should use default status of UNKNOWN")
        void shouldUseDefaultStatusOfUnknown() {
            RecognitionResult result = RecognitionResult.builder().build();

            assertThat(result.getStatus()).isEqualTo(RecognitionResult.Status.UNKNOWN);
        }
    }

    @Nested
    @DisplayName("Status Handling")
    class StatusHandling {

        @ParameterizedTest
        @DisplayName("Should handle all status values")
        @EnumSource(RecognitionResult.Status.class)
        void shouldHandleAllStatusValues(RecognitionResult.Status status) {
            RecognitionResult result = RecognitionResult.builder()
                .status(status)
                .build();

            assertThat(result.getStatus()).isEqualTo(status);
        }

        @Test
        @DisplayName("Should identify recognized status")
        void shouldIdentifyRecognizedStatus() {
            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .build();

            assertThat(result.isRecognized()).isTrue();
            assertThat(result.isFaceDetected()).isTrue();
            assertThat(result.isError()).isFalse();
        }

        @Test
        @DisplayName("Should identify unknown status")
        void shouldIdentifyUnknownStatus() {
            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.UNKNOWN)
                .build();

            assertThat(result.isRecognized()).isFalse();
            assertThat(result.isFaceDetected()).isTrue();
            assertThat(result.isError()).isFalse();
        }

        @Test
        @DisplayName("Should identify no face detected status")
        void shouldIdentifyNoFaceDetectedStatus() {
            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.NO_FACE_DETECTED)
                .build();

            assertThat(result.isRecognized()).isFalse();
            assertThat(result.isFaceDetected()).isFalse();
            assertThat(result.isError()).isFalse();
        }

        @Test
        @DisplayName("Should identify error status")
        void shouldIdentifyErrorStatus() {
            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.ERROR)
                .build();

            assertThat(result.isRecognized()).isFalse();
            assertThat(result.isFaceDetected()).isFalse();
            assertThat(result.isError()).isTrue();
        }

        @Test
        @DisplayName("Should identify multiple faces status")
        void shouldIdentifyMultipleFacesStatus() {
            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.MULTIPLE_FACES)
                .build();

            assertThat(result.isRecognized()).isFalse();
            assertThat(result.isFaceDetected()).isFalse();
        }

        @Test
        @DisplayName("Should identify poor quality status")
        void shouldIdentifyPoorQualityStatus() {
            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.POOR_QUALITY)
                .build();

            assertThat(result.isRecognized()).isFalse();
            assertThat(result.isFaceDetected()).isFalse();
        }
    }

    @Nested
    @DisplayName("Alternatives Handling")
    class AlternativesHandling {

        @Test
        @DisplayName("Should return empty alternatives when none set")
        void shouldReturnEmptyAlternativesWhenNoneSet() {
            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .build();

            assertThat(result.getAlternatives()).isEmpty();
        }

        @Test
        @DisplayName("Should return unmodifiable alternatives list")
        void shouldReturnUnmodifiableAlternativesList() {
            Identity identity = createIdentity("Person");
            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .addAlternative(new RecognitionResult.MatchResult(identity, 0.8, 0.1))
                .build();

            List<RecognitionResult.MatchResult> alternatives = result.getAlternatives();

            assertThatThrownBy(() -> alternatives.add(null))
                .isInstanceOf(UnsupportedOperationException.class);
        }

        @Test
        @DisplayName("Should get top N alternatives")
        void shouldGetTopNAlternatives() {
            Identity id1 = createIdentity("Person1");
            Identity id2 = createIdentity("Person2");
            Identity id3 = createIdentity("Person3");

            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.UNKNOWN)
                .addAlternative(new RecognitionResult.MatchResult(id1, 0.6, 0.3))
                .addAlternative(new RecognitionResult.MatchResult(id2, 0.9, 0.05))
                .addAlternative(new RecognitionResult.MatchResult(id3, 0.7, 0.2))
                .build();

            List<RecognitionResult.MatchResult> top2 = result.getTopAlternatives(2);

            assertThat(top2).hasSize(2);
            // Should be sorted by confidence (highest first)
            assertThat(top2.get(0).getConfidence()).isEqualTo(0.9);
            assertThat(top2.get(1).getConfidence()).isEqualTo(0.7);
        }

        @Test
        @DisplayName("Should handle requesting more alternatives than available")
        void shouldHandleRequestingMoreAlternativesThanAvailable() {
            Identity identity = createIdentity("Person");

            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.UNKNOWN)
                .addAlternative(new RecognitionResult.MatchResult(identity, 0.8, 0.1))
                .build();

            List<RecognitionResult.MatchResult> top5 = result.getTopAlternatives(5);

            assertThat(top5).hasSize(1);
        }
    }

    @Nested
    @DisplayName("Factory Methods")
    class FactoryMethods {

        @Test
        @DisplayName("Should create recognized result")
        void shouldCreateRecognizedResult() {
            Identity identity = createIdentity("John Doe");

            RecognitionResult result = RecognitionResult.recognized(identity, 0.95, 0.05);

            assertThat(result.getStatus()).isEqualTo(RecognitionResult.Status.RECOGNIZED);
            assertThat(result.isRecognized()).isTrue();
            assertThat(result.getBestMatch()).isPresent();
            assertThat(result.getConfidence()).isEqualTo(0.95);
            assertThat(result.getDistance()).isEqualTo(0.05);
            assertThat(result.getIdentity()).isPresent();
            assertThat(result.getIdentity().get()).isEqualTo(identity);
        }

        @Test
        @DisplayName("Should create unknown result")
        void shouldCreateUnknownResult() {
            RecognitionResult result = RecognitionResult.unknown();

            assertThat(result.getStatus()).isEqualTo(RecognitionResult.Status.UNKNOWN);
            assertThat(result.isRecognized()).isFalse();
            assertThat(result.isFaceDetected()).isTrue();
            assertThat(result.getBestMatch()).isEmpty();
            assertThat(result.getConfidence()).isEqualTo(0.0);
            assertThat(result.getDistance()).isEqualTo(Double.MAX_VALUE);
        }

        @Test
        @DisplayName("Should create no face detected result")
        void shouldCreateNoFaceDetectedResult() {
            RecognitionResult result = RecognitionResult.noFaceDetected();

            assertThat(result.getStatus()).isEqualTo(RecognitionResult.Status.NO_FACE_DETECTED);
            assertThat(result.isRecognized()).isFalse();
            assertThat(result.isFaceDetected()).isFalse();
        }

        @Test
        @DisplayName("Should create error result")
        void shouldCreateErrorResult() {
            RecognitionResult result = RecognitionResult.error("Test error message");

            assertThat(result.getStatus()).isEqualTo(RecognitionResult.Status.ERROR);
            assertThat(result.isError()).isTrue();
            assertThat(result.getErrorMessage()).isPresent();
            assertThat(result.getErrorMessage().get()).isEqualTo("Test error message");
        }
    }

    @Nested
    @DisplayName("Convenience Getters")
    class ConvenienceGetters {

        @Test
        @DisplayName("Should get identity from best match")
        void shouldGetIdentityFromBestMatch() {
            Identity identity = createIdentity("John Doe");

            RecognitionResult result = RecognitionResult.recognized(identity, 0.95, 0.05);

            Optional<Identity> retrieved = result.getIdentity();

            assertThat(retrieved).isPresent();
            assertThat(retrieved.get().getName()).isEqualTo("John Doe");
        }

        @Test
        @DisplayName("Should return empty identity when no match")
        void shouldReturnEmptyIdentityWhenNoMatch() {
            RecognitionResult result = RecognitionResult.unknown();

            assertThat(result.getIdentity()).isEmpty();
        }

        @Test
        @DisplayName("Should return 0.0 confidence when no match")
        void shouldReturnZeroConfidenceWhenNoMatch() {
            RecognitionResult result = RecognitionResult.unknown();

            assertThat(result.getConfidence()).isEqualTo(0.0);
        }

        @Test
        @DisplayName("Should return MAX_VALUE distance when no match")
        void shouldReturnMaxValueDistanceWhenNoMatch() {
            RecognitionResult result = RecognitionResult.unknown();

            assertThat(result.getDistance()).isEqualTo(Double.MAX_VALUE);
        }

        @Test
        @DisplayName("Should return empty optional for unset detected face")
        void shouldReturnEmptyOptionalForUnsetDetectedFace() {
            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .build();

            assertThat(result.getDetectedFace()).isEmpty();
        }

        @Test
        @DisplayName("Should return empty optional for unset features")
        void shouldReturnEmptyOptionalForUnsetFeatures() {
            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .build();

            assertThat(result.getExtractedFeatures()).isEmpty();
        }

        @Test
        @DisplayName("Should return empty optional for unset metrics")
        void shouldReturnEmptyOptionalForUnsetMetrics() {
            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .build();

            assertThat(result.getMetrics()).isEmpty();
        }
    }

    @Nested
    @DisplayName("MatchResult Tests")
    class MatchResultTests {

        @Test
        @DisplayName("Should create match result")
        void shouldCreateMatchResult() {
            Identity identity = createIdentity("Test Person");

            RecognitionResult.MatchResult match = new RecognitionResult.MatchResult(identity, 0.95, 0.05);

            assertThat(match.getIdentity()).isEqualTo(identity);
            assertThat(match.getConfidence()).isEqualTo(0.95);
            assertThat(match.getDistance()).isEqualTo(0.05);
        }

        @Test
        @DisplayName("Should throw exception for null identity")
        void shouldThrowExceptionForNullIdentity() {
            assertThatNullPointerException()
                .isThrownBy(() -> new RecognitionResult.MatchResult(null, 0.95, 0.05));
        }

        @Test
        @DisplayName("Should compare match results by confidence")
        void shouldCompareMatchResultsByConfidence() {
            Identity id1 = createIdentity("Person1");
            Identity id2 = createIdentity("Person2");

            RecognitionResult.MatchResult match1 = new RecognitionResult.MatchResult(id1, 0.9, 0.1);
            RecognitionResult.MatchResult match2 = new RecognitionResult.MatchResult(id2, 0.8, 0.2);

            // Higher confidence should come first (negative comparison result)
            assertThat(match1.compareTo(match2)).isNegative();
            assertThat(match2.compareTo(match1)).isPositive();
            assertThat(match1.compareTo(match1)).isZero();
        }

        @Test
        @DisplayName("Should format match result toString correctly")
        void shouldFormatMatchResultToStringCorrectly() {
            Identity identity = createIdentity("John Doe");

            RecognitionResult.MatchResult match = new RecognitionResult.MatchResult(identity, 0.95, 0.05);
            String str = match.toString();

            assertThat(str).contains("Match");
            assertThat(str).contains("John Doe");
            assertThat(str).contains("0.950");
            assertThat(str).contains("0.0500");
        }
    }

    @Nested
    @DisplayName("ProcessingMetrics Tests")
    class ProcessingMetricsTests {

        @Test
        @DisplayName("Should create processing metrics")
        void shouldCreateProcessingMetrics() {
            RecognitionResult.ProcessingMetrics metrics =
                new RecognitionResult.ProcessingMetrics(10, 20, 30, 60);

            assertThat(metrics.getDetectionTimeMs()).isEqualTo(10);
            assertThat(metrics.getExtractionTimeMs()).isEqualTo(20);
            assertThat(metrics.getMatchingTimeMs()).isEqualTo(30);
            assertThat(metrics.getTotalTimeMs()).isEqualTo(60);
        }

        @Test
        @DisplayName("Should format processing metrics toString correctly")
        void shouldFormatProcessingMetricsToStringCorrectly() {
            RecognitionResult.ProcessingMetrics metrics =
                new RecognitionResult.ProcessingMetrics(10, 20, 30, 60);
            String str = metrics.toString();

            assertThat(str).contains("Metrics");
            assertThat(str).contains("total=60ms");
            assertThat(str).contains("detect=10ms");
            assertThat(str).contains("extract=20ms");
            assertThat(str).contains("match=30ms");
        }
    }

    @Nested
    @DisplayName("ToString Tests")
    class ToStringTests {

        @Test
        @DisplayName("Should format recognized result toString")
        void shouldFormatRecognizedResultToString() {
            Identity identity = createIdentity("John Doe");
            RecognitionResult.ProcessingMetrics metrics =
                new RecognitionResult.ProcessingMetrics(10, 20, 30, 60);

            RecognitionResult result = RecognitionResult.builder()
                .status(RecognitionResult.Status.RECOGNIZED)
                .bestMatch(new RecognitionResult.MatchResult(identity, 0.95, 0.05))
                .metrics(metrics)
                .build();

            String str = result.toString();

            assertThat(str).contains("RecognitionResult");
            assertThat(str).contains("RECOGNIZED");
            assertThat(str).contains("John Doe");
            assertThat(str).contains("0.950");
            assertThat(str).contains("60ms");
        }

        @Test
        @DisplayName("Should format unknown result toString")
        void shouldFormatUnknownResultToString() {
            RecognitionResult result = RecognitionResult.unknown();
            String str = result.toString();

            assertThat(str).contains("RecognitionResult");
            assertThat(str).contains("UNKNOWN");
        }
    }

    // Helper methods

    private FeatureVector createTestFeatureVector() {
        double[] features = new double[128];
        for (int i = 0; i < features.length; i++) {
            features[i] = Math.random();
        }
        return new FeatureVector(features, "test", 1);
    }

    private Identity createIdentity(String name) {
        Identity identity = new Identity(name);
        identity.enrollSample(createTestFeatureVector(), 0.9, "test-source");
        return identity;
    }
}
