package com.facerecognition.unit.domain;

import com.facerecognition.domain.model.FeatureVector;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for FeatureVector class.
 */
@DisplayName("FeatureVector Tests")
class FeatureVectorTest {

    private static final double DELTA = 1e-10;

    @Nested
    @DisplayName("Construction")
    class ConstructionTests {

        @Test
        @DisplayName("Creates feature vector with valid array")
        void createsWithValidArray() {
            double[] features = {1.0, 2.0, 3.0};
            FeatureVector fv = new FeatureVector(features);

            assertEquals(3, fv.getDimension());
            assertArrayEquals(features, fv.getFeatures(), DELTA);
        }

        @Test
        @DisplayName("Creates with algorithm metadata")
        void createsWithMetadata() {
            double[] features = {1.0, 2.0};
            FeatureVector fv = new FeatureVector(features, "Eigenfaces", 2);

            assertEquals("Eigenfaces", fv.getAlgorithmName());
            assertEquals(2, fv.getVersion());
        }

        @Test
        @DisplayName("Throws on null features")
        void throwsOnNullFeatures() {
            assertThrows(NullPointerException.class, () -> new FeatureVector(null));
        }

        @Test
        @DisplayName("Throws on empty features")
        void throwsOnEmptyFeatures() {
            assertThrows(IllegalArgumentException.class, () -> new FeatureVector(new double[0]));
        }

        @Test
        @DisplayName("Returns copy of features, not original")
        void returnsCopyOfFeatures() {
            double[] original = {1.0, 2.0, 3.0};
            FeatureVector fv = new FeatureVector(original);

            double[] retrieved = fv.getFeatures();
            retrieved[0] = 999.0;

            assertEquals(1.0, fv.getFeature(0), DELTA);
        }
    }

    @Nested
    @DisplayName("Distance Metrics")
    class DistanceTests {

        @Test
        @DisplayName("Euclidean distance - identical vectors")
        void euclideanDistanceIdentical() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0, 3.0});
            FeatureVector v2 = new FeatureVector(new double[]{1.0, 2.0, 3.0});

            assertEquals(0.0, v1.euclideanDistance(v2), DELTA);
        }

        @Test
        @DisplayName("Euclidean distance - different vectors")
        void euclideanDistanceDifferent() {
            FeatureVector v1 = new FeatureVector(new double[]{0.0, 0.0, 0.0});
            FeatureVector v2 = new FeatureVector(new double[]{3.0, 4.0, 0.0});

            // sqrt(3^2 + 4^2) = 5
            assertEquals(5.0, v1.euclideanDistance(v2), DELTA);
        }

        @Test
        @DisplayName("Cosine similarity - identical vectors")
        void cosineSimilarityIdentical() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0, 3.0});
            FeatureVector v2 = new FeatureVector(new double[]{1.0, 2.0, 3.0});

            assertEquals(1.0, v1.cosineSimilarity(v2), DELTA);
        }

        @Test
        @DisplayName("Cosine similarity - orthogonal vectors")
        void cosineSimilarityOrthogonal() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 0.0});
            FeatureVector v2 = new FeatureVector(new double[]{0.0, 1.0});

            assertEquals(0.0, v1.cosineSimilarity(v2), DELTA);
        }

        @Test
        @DisplayName("Cosine similarity - opposite vectors")
        void cosineSimilarityOpposite() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 0.0});
            FeatureVector v2 = new FeatureVector(new double[]{-1.0, 0.0});

            assertEquals(-1.0, v1.cosineSimilarity(v2), DELTA);
        }

        @Test
        @DisplayName("Cosine distance from similarity")
        void cosineDistanceFromSimilarity() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0, 3.0});
            FeatureVector v2 = new FeatureVector(new double[]{1.0, 2.0, 3.0});

            assertEquals(0.0, v1.cosineDistance(v2), DELTA);
        }

        @Test
        @DisplayName("Manhattan distance")
        void manhattanDistance() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0, 3.0});
            FeatureVector v2 = new FeatureVector(new double[]{4.0, 6.0, 8.0});

            // |4-1| + |6-2| + |8-3| = 3 + 4 + 5 = 12
            assertEquals(12.0, v1.manhattanDistance(v2), DELTA);
        }

        @Test
        @DisplayName("Chi-square distance")
        void chiSquareDistance() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0, 3.0});
            FeatureVector v2 = new FeatureVector(new double[]{1.0, 2.0, 3.0});

            assertEquals(0.0, v1.chiSquareDistance(v2), DELTA);
        }

        @Test
        @DisplayName("Throws on dimension mismatch")
        void throwsOnDimensionMismatch() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0});
            FeatureVector v2 = new FeatureVector(new double[]{1.0, 2.0, 3.0});

            assertThrows(IllegalArgumentException.class, () -> v1.euclideanDistance(v2));
        }
    }

    @Nested
    @DisplayName("Vector Operations")
    class OperationsTests {

        @Test
        @DisplayName("Computes norm correctly")
        void computesNorm() {
            FeatureVector v = new FeatureVector(new double[]{3.0, 4.0});
            assertEquals(5.0, v.norm(), DELTA);
        }

        @Test
        @DisplayName("Normalizes vector")
        void normalizesVector() {
            FeatureVector v = new FeatureVector(new double[]{3.0, 4.0});
            FeatureVector normalized = v.normalize();

            assertEquals(1.0, normalized.norm(), DELTA);
            assertEquals(0.6, normalized.getFeature(0), DELTA);
            assertEquals(0.8, normalized.getFeature(1), DELTA);
        }

        @Test
        @DisplayName("Adds vectors")
        void addsVectors() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0});
            FeatureVector v2 = new FeatureVector(new double[]{3.0, 4.0});
            FeatureVector sum = v1.add(v2);

            assertEquals(4.0, sum.getFeature(0), DELTA);
            assertEquals(6.0, sum.getFeature(1), DELTA);
        }

        @Test
        @DisplayName("Subtracts vectors")
        void subtractsVectors() {
            FeatureVector v1 = new FeatureVector(new double[]{5.0, 7.0});
            FeatureVector v2 = new FeatureVector(new double[]{2.0, 3.0});
            FeatureVector diff = v1.subtract(v2);

            assertEquals(3.0, diff.getFeature(0), DELTA);
            assertEquals(4.0, diff.getFeature(1), DELTA);
        }

        @Test
        @DisplayName("Scales vector")
        void scalesVector() {
            FeatureVector v = new FeatureVector(new double[]{2.0, 3.0});
            FeatureVector scaled = v.scale(2.0);

            assertEquals(4.0, scaled.getFeature(0), DELTA);
            assertEquals(6.0, scaled.getFeature(1), DELTA);
        }

        @Test
        @DisplayName("Computes dot product")
        void computesDotProduct() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0, 3.0});
            FeatureVector v2 = new FeatureVector(new double[]{4.0, 5.0, 6.0});

            // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
            assertEquals(32.0, v1.dot(v2), DELTA);
        }
    }

    @Nested
    @DisplayName("Compatibility")
    class CompatibilityTests {

        @Test
        @DisplayName("Compatible vectors")
        void compatibleVectors() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0}, "Eigenfaces", 1);
            FeatureVector v2 = new FeatureVector(new double[]{3.0, 4.0}, "Eigenfaces", 1);

            assertTrue(v1.isCompatibleWith(v2));
        }

        @Test
        @DisplayName("Incompatible - different dimensions")
        void incompatibleDimensions() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0}, "Eigenfaces", 1);
            FeatureVector v2 = new FeatureVector(new double[]{3.0, 4.0, 5.0}, "Eigenfaces", 1);

            assertFalse(v1.isCompatibleWith(v2));
        }

        @Test
        @DisplayName("Incompatible - different algorithms")
        void incompatibleAlgorithms() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0}, "Eigenfaces", 1);
            FeatureVector v2 = new FeatureVector(new double[]{3.0, 4.0}, "LBPH", 1);

            assertFalse(v1.isCompatibleWith(v2));
        }
    }

    @Nested
    @DisplayName("Equality and Hashing")
    class EqualityTests {

        @Test
        @DisplayName("Equal vectors are equal")
        void equalVectorsAreEqual() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0, 3.0});
            FeatureVector v2 = new FeatureVector(new double[]{1.0, 2.0, 3.0});

            assertEquals(v1, v2);
            assertEquals(v1.hashCode(), v2.hashCode());
        }

        @Test
        @DisplayName("Different vectors are not equal")
        void differentVectorsNotEqual() {
            FeatureVector v1 = new FeatureVector(new double[]{1.0, 2.0, 3.0});
            FeatureVector v2 = new FeatureVector(new double[]{1.0, 2.0, 4.0});

            assertNotEquals(v1, v2);
        }
    }
}
