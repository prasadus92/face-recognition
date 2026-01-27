package com.facerecognition.domain.model;

import org.junit.jupiter.api.*;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.awt.Rectangle;

import static org.assertj.core.api.Assertions.*;

/**
 * Comprehensive unit tests for FaceRegion domain model.
 */
@DisplayName("FaceRegion Tests")
class FaceRegionTest {

    @Nested
    @DisplayName("IoU (Intersection over Union) Calculation")
    class IoUCalculation {

        @Test
        @DisplayName("Should return 1.0 for identical regions")
        void shouldReturnOneForIdenticalRegions() {
            FaceRegion region1 = new FaceRegion(10, 10, 100, 100, 0.9);
            FaceRegion region2 = new FaceRegion(10, 10, 100, 100, 0.8);

            double iou = region1.intersectionOverUnion(region2);

            assertThat(iou).isCloseTo(1.0, within(0.0001));
        }

        @Test
        @DisplayName("Should return 0.0 for non-overlapping regions")
        void shouldReturnZeroForNonOverlappingRegions() {
            FaceRegion region1 = new FaceRegion(0, 0, 50, 50, 0.9);
            FaceRegion region2 = new FaceRegion(100, 100, 50, 50, 0.8);

            double iou = region1.intersectionOverUnion(region2);

            assertThat(iou).isEqualTo(0.0);
        }

        @Test
        @DisplayName("Should calculate correct IoU for partially overlapping regions")
        void shouldCalculateCorrectIoUForPartialOverlap() {
            // Two 100x100 regions, overlapping by 50x100
            FaceRegion region1 = new FaceRegion(0, 0, 100, 100, 0.9);
            FaceRegion region2 = new FaceRegion(50, 0, 100, 100, 0.8);

            double iou = region1.intersectionOverUnion(region2);

            // Intersection: 50 * 100 = 5000
            // Union: 100*100 + 100*100 - 5000 = 15000
            // IoU: 5000 / 15000 = 0.333...
            assertThat(iou).isCloseTo(0.333, within(0.01));
        }

        @Test
        @DisplayName("Should return 0.0 when regions share only an edge")
        void shouldReturnZeroWhenSharingOnlyEdge() {
            FaceRegion region1 = new FaceRegion(0, 0, 50, 50, 0.9);
            FaceRegion region2 = new FaceRegion(50, 0, 50, 50, 0.8);

            double iou = region1.intersectionOverUnion(region2);

            assertThat(iou).isEqualTo(0.0);
        }

        @Test
        @DisplayName("Should handle region completely inside another")
        void shouldHandleRegionCompletelyInsideAnother() {
            FaceRegion outer = new FaceRegion(0, 0, 100, 100, 0.9);
            FaceRegion inner = new FaceRegion(25, 25, 50, 50, 0.8);

            double iou = outer.intersectionOverUnion(inner);

            // Intersection: 50 * 50 = 2500
            // Union: 100*100 + 50*50 - 2500 = 10000
            // IoU: 2500 / 10000 = 0.25
            assertThat(iou).isCloseTo(0.25, within(0.01));
        }

        @ParameterizedTest
        @DisplayName("Should calculate IoU for various overlap scenarios")
        @CsvSource({
            "0, 0, 100, 100, 0, 0, 100, 100, 1.0",      // Identical
            "0, 0, 100, 100, 200, 0, 100, 100, 0.0",    // No overlap
            "0, 0, 100, 100, 50, 50, 100, 100, 0.142",  // 50x50 overlap
        })
        void shouldCalculateIoUForVariousScenarios(
                int x1, int y1, int w1, int h1,
                int x2, int y2, int w2, int h2,
                double expectedIoU) {
            FaceRegion region1 = new FaceRegion(x1, y1, w1, h1, 0.9);
            FaceRegion region2 = new FaceRegion(x2, y2, w2, h2, 0.8);

            double iou = region1.intersectionOverUnion(region2);

            assertThat(iou).isCloseTo(expectedIoU, within(0.01));
        }
    }

    @Nested
    @DisplayName("Contains Method")
    class ContainsMethod {

        @Nested
        @DisplayName("Point Containment")
        class PointContainment {

            @Test
            @DisplayName("Should contain point inside region")
            void shouldContainPointInsideRegion() {
                FaceRegion region = new FaceRegion(10, 10, 100, 100, 0.9);

                assertThat(region.contains(50, 50)).isTrue();
            }

            @Test
            @DisplayName("Should contain point at top-left corner")
            void shouldContainPointAtTopLeftCorner() {
                FaceRegion region = new FaceRegion(10, 10, 100, 100, 0.9);

                assertThat(region.contains(10, 10)).isTrue();
            }

            @Test
            @DisplayName("Should not contain point at bottom-right boundary")
            void shouldNotContainPointAtBottomRightBoundary() {
                FaceRegion region = new FaceRegion(10, 10, 100, 100, 0.9);

                assertThat(region.contains(110, 110)).isFalse();
            }

            @Test
            @DisplayName("Should not contain point outside region")
            void shouldNotContainPointOutsideRegion() {
                FaceRegion region = new FaceRegion(10, 10, 100, 100, 0.9);

                assertThat(region.contains(0, 0)).isFalse();
                assertThat(region.contains(200, 200)).isFalse();
            }

            @Test
            @DisplayName("Should contain point just inside right edge")
            void shouldContainPointJustInsideRightEdge() {
                FaceRegion region = new FaceRegion(10, 10, 100, 100, 0.9);

                assertThat(region.contains(109, 50)).isTrue();
            }
        }

        @Nested
        @DisplayName("Region Containment")
        class RegionContainment {

            @Test
            @DisplayName("Should contain smaller region inside")
            void shouldContainSmallerRegionInside() {
                FaceRegion outer = new FaceRegion(0, 0, 100, 100, 0.9);
                FaceRegion inner = new FaceRegion(25, 25, 50, 50, 0.8);

                assertThat(outer.contains(inner)).isTrue();
            }

            @Test
            @DisplayName("Should contain identical region")
            void shouldContainIdenticalRegion() {
                FaceRegion region1 = new FaceRegion(10, 10, 100, 100, 0.9);
                FaceRegion region2 = new FaceRegion(10, 10, 100, 100, 0.8);

                assertThat(region1.contains(region2)).isTrue();
            }

            @Test
            @DisplayName("Should not contain larger region")
            void shouldNotContainLargerRegion() {
                FaceRegion smaller = new FaceRegion(25, 25, 50, 50, 0.9);
                FaceRegion larger = new FaceRegion(0, 0, 100, 100, 0.8);

                assertThat(smaller.contains(larger)).isFalse();
            }

            @Test
            @DisplayName("Should not contain partially overlapping region")
            void shouldNotContainPartiallyOverlappingRegion() {
                FaceRegion region1 = new FaceRegion(0, 0, 100, 100, 0.9);
                FaceRegion region2 = new FaceRegion(50, 50, 100, 100, 0.8);

                assertThat(region1.contains(region2)).isFalse();
            }

            @Test
            @DisplayName("Should contain region touching inner edges")
            void shouldContainRegionTouchingInnerEdges() {
                FaceRegion outer = new FaceRegion(0, 0, 100, 100, 0.9);
                FaceRegion inner = new FaceRegion(0, 0, 100, 100, 0.8);

                assertThat(outer.contains(inner)).isTrue();
            }
        }
    }

    @Nested
    @DisplayName("Expansion and Shrinking")
    class ExpansionAndShrinking {

        @Test
        @DisplayName("Should expand region by factor")
        void shouldExpandRegionByFactor() {
            FaceRegion original = new FaceRegion(50, 50, 100, 100, 0.9);

            FaceRegion expanded = original.expand(1.5);

            assertThat(expanded.getWidth()).isEqualTo(150);
            assertThat(expanded.getHeight()).isEqualTo(150);
            // Center should remain the same
            assertThat(expanded.getCenterX()).isEqualTo(original.getCenterX());
            assertThat(expanded.getCenterY()).isEqualTo(original.getCenterY());
        }

        @Test
        @DisplayName("Should shrink region by factor less than 1")
        void shouldShrinkRegionByFactorLessThanOne() {
            FaceRegion original = new FaceRegion(50, 50, 100, 100, 0.9);

            FaceRegion shrunk = original.expand(0.5);

            assertThat(shrunk.getWidth()).isEqualTo(50);
            assertThat(shrunk.getHeight()).isEqualTo(50);
        }

        @Test
        @DisplayName("Should preserve confidence after expansion")
        void shouldPreserveConfidenceAfterExpansion() {
            FaceRegion original = new FaceRegion(50, 50, 100, 100, 0.9);

            FaceRegion expanded = original.expand(1.5);

            assertThat(expanded.getConfidence()).isEqualTo(0.9);
        }

        @Test
        @DisplayName("Should return same size for factor 1.0")
        void shouldReturnSameSizeForFactorOne() {
            FaceRegion original = new FaceRegion(50, 50, 100, 100, 0.9);

            FaceRegion same = original.expand(1.0);

            assertThat(same.getWidth()).isEqualTo(original.getWidth());
            assertThat(same.getHeight()).isEqualTo(original.getHeight());
        }

        @Test
        @DisplayName("Should expand asymmetric region correctly")
        void shouldExpandAsymmetricRegionCorrectly() {
            FaceRegion original = new FaceRegion(50, 50, 100, 200, 0.9);

            FaceRegion expanded = original.expand(2.0);

            assertThat(expanded.getWidth()).isEqualTo(200);
            assertThat(expanded.getHeight()).isEqualTo(400);
        }

        @Test
        @DisplayName("Should calculate correct position after expansion")
        void shouldCalculateCorrectPositionAfterExpansion() {
            FaceRegion original = new FaceRegion(100, 100, 100, 100, 0.9);
            // Center at (150, 150)

            FaceRegion expanded = original.expand(2.0);
            // New size: 200x200, center should remain at (150, 150)
            // New position: (150 - 100, 150 - 100) = (50, 50)

            assertThat(expanded.getX()).isEqualTo(50);
            assertThat(expanded.getY()).isEqualTo(50);
        }
    }

    @Nested
    @DisplayName("Intersects Tests")
    class IntersectsTests {

        @Test
        @DisplayName("Should detect overlapping regions intersect")
        void shouldDetectOverlappingRegionsIntersect() {
            FaceRegion region1 = new FaceRegion(0, 0, 100, 100, 0.9);
            FaceRegion region2 = new FaceRegion(50, 50, 100, 100, 0.8);

            double iou = region1.intersectionOverUnion(region2);

            assertThat(iou).isGreaterThan(0.0);
        }

        @Test
        @DisplayName("Should detect non-overlapping regions do not intersect")
        void shouldDetectNonOverlappingRegionsDoNotIntersect() {
            FaceRegion region1 = new FaceRegion(0, 0, 50, 50, 0.9);
            FaceRegion region2 = new FaceRegion(100, 100, 50, 50, 0.8);

            double iou = region1.intersectionOverUnion(region2);

            assertThat(iou).isEqualTo(0.0);
        }

        @Test
        @DisplayName("Should return non-zero IoU for any intersection")
        void shouldReturnNonZeroIoUForAnyIntersection() {
            FaceRegion region1 = new FaceRegion(0, 0, 100, 100, 0.9);
            FaceRegion region2 = new FaceRegion(99, 99, 100, 100, 0.8);

            double iou = region1.intersectionOverUnion(region2);

            assertThat(iou).isGreaterThan(0.0);
        }
    }

    @Nested
    @DisplayName("Construction and Validation")
    class ConstructionAndValidation {

        @Test
        @DisplayName("Should create region with valid parameters")
        void shouldCreateRegionWithValidParameters() {
            FaceRegion region = new FaceRegion(10, 20, 100, 150, 0.95);

            assertThat(region.getX()).isEqualTo(10);
            assertThat(region.getY()).isEqualTo(20);
            assertThat(region.getWidth()).isEqualTo(100);
            assertThat(region.getHeight()).isEqualTo(150);
            assertThat(region.getConfidence()).isEqualTo(0.95);
        }

        @Test
        @DisplayName("Should throw exception for zero width")
        void shouldThrowExceptionForZeroWidth() {
            assertThatIllegalArgumentException()
                .isThrownBy(() -> new FaceRegion(10, 10, 0, 100, 0.9))
                .withMessageContaining("Width and height must be positive");
        }

        @Test
        @DisplayName("Should throw exception for zero height")
        void shouldThrowExceptionForZeroHeight() {
            assertThatIllegalArgumentException()
                .isThrownBy(() -> new FaceRegion(10, 10, 100, 0, 0.9))
                .withMessageContaining("Width and height must be positive");
        }

        @Test
        @DisplayName("Should throw exception for negative width")
        void shouldThrowExceptionForNegativeWidth() {
            assertThatIllegalArgumentException()
                .isThrownBy(() -> new FaceRegion(10, 10, -100, 100, 0.9))
                .withMessageContaining("Width and height must be positive");
        }

        @Test
        @DisplayName("Should throw exception for negative height")
        void shouldThrowExceptionForNegativeHeight() {
            assertThatIllegalArgumentException()
                .isThrownBy(() -> new FaceRegion(10, 10, 100, -100, 0.9))
                .withMessageContaining("Width and height must be positive");
        }

        @ParameterizedTest
        @DisplayName("Should throw exception for invalid confidence")
        @ValueSource(doubles = {-0.1, -1.0, 1.1, 2.0})
        void shouldThrowExceptionForInvalidConfidence(double confidence) {
            assertThatIllegalArgumentException()
                .isThrownBy(() -> new FaceRegion(10, 10, 100, 100, confidence))
                .withMessageContaining("Confidence must be between 0.0 and 1.0");
        }

        @ParameterizedTest
        @DisplayName("Should accept valid confidence values")
        @ValueSource(doubles = {0.0, 0.5, 1.0, 0.001, 0.999})
        void shouldAcceptValidConfidenceValues(double confidence) {
            FaceRegion region = new FaceRegion(10, 10, 100, 100, confidence);

            assertThat(region.getConfidence()).isEqualTo(confidence);
        }

        @Test
        @DisplayName("Should allow negative x and y coordinates")
        void shouldAllowNegativeCoordinates() {
            FaceRegion region = new FaceRegion(-50, -30, 100, 100, 0.9);

            assertThat(region.getX()).isEqualTo(-50);
            assertThat(region.getY()).isEqualTo(-30);
        }
    }

    @Nested
    @DisplayName("Geometric Properties")
    class GeometricProperties {

        @Test
        @DisplayName("Should calculate area correctly")
        void shouldCalculateAreaCorrectly() {
            FaceRegion region = new FaceRegion(0, 0, 100, 200, 0.9);

            assertThat(region.getArea()).isEqualTo(20000);
        }

        @Test
        @DisplayName("Should calculate center X correctly")
        void shouldCalculateCenterXCorrectly() {
            FaceRegion region = new FaceRegion(100, 50, 200, 100, 0.9);

            assertThat(region.getCenterX()).isEqualTo(200);
        }

        @Test
        @DisplayName("Should calculate center Y correctly")
        void shouldCalculateCenterYCorrectly() {
            FaceRegion region = new FaceRegion(100, 50, 200, 100, 0.9);

            assertThat(region.getCenterY()).isEqualTo(100);
        }

        @Test
        @DisplayName("Should calculate aspect ratio correctly")
        void shouldCalculateAspectRatioCorrectly() {
            FaceRegion region = new FaceRegion(0, 0, 200, 100, 0.9);

            assertThat(region.getAspectRatio()).isEqualTo(2.0);
        }

        @Test
        @DisplayName("Should convert to Rectangle")
        void shouldConvertToRectangle() {
            FaceRegion region = new FaceRegion(10, 20, 100, 150, 0.9);

            Rectangle rect = region.toRectangle();

            assertThat(rect.x).isEqualTo(10);
            assertThat(rect.y).isEqualTo(20);
            assertThat(rect.width).isEqualTo(100);
            assertThat(rect.height).isEqualTo(150);
        }

        @Test
        @DisplayName("Should create from Rectangle")
        void shouldCreateFromRectangle() {
            Rectangle rect = new Rectangle(10, 20, 100, 150);

            FaceRegion region = FaceRegion.fromRectangle(rect, 0.85);

            assertThat(region.getX()).isEqualTo(10);
            assertThat(region.getY()).isEqualTo(20);
            assertThat(region.getWidth()).isEqualTo(100);
            assertThat(region.getHeight()).isEqualTo(150);
            assertThat(region.getConfidence()).isEqualTo(0.85);
        }
    }

    @Nested
    @DisplayName("Landmarks")
    class LandmarksTests {

        @Test
        @DisplayName("Should not have landmarks by default")
        void shouldNotHaveLandmarksByDefault() {
            FaceRegion region = new FaceRegion(10, 10, 100, 100, 0.9);

            assertThat(region.hasLandmarks()).isFalse();
            assertThat(region.getLandmarks()).isNull();
        }

        @Test
        @DisplayName("Should create region with landmarks")
        void shouldCreateRegionWithLandmarks() {
            FaceLandmarks landmarks = createMockLandmarks();
            FaceRegion region = new FaceRegion(10, 10, 100, 100, 0.9, landmarks);

            assertThat(region.hasLandmarks()).isTrue();
            assertThat(region.getLandmarks()).isEqualTo(landmarks);
        }

        @Test
        @DisplayName("Should add landmarks with withLandmarks method")
        void shouldAddLandmarksWithMethod() {
            FaceRegion original = new FaceRegion(10, 10, 100, 100, 0.9);
            FaceLandmarks landmarks = createMockLandmarks();

            FaceRegion withLandmarks = original.withLandmarks(landmarks);

            assertThat(withLandmarks.hasLandmarks()).isTrue();
            assertThat(withLandmarks.getLandmarks()).isEqualTo(landmarks);
            assertThat(withLandmarks.getX()).isEqualTo(original.getX());
            assertThat(withLandmarks.getY()).isEqualTo(original.getY());
        }

        private FaceLandmarks createMockLandmarks() {
            // Create a simple landmarks object for testing
            return FaceLandmarks.create5Point(
                new java.awt.Point(30, 30),  // leftEye
                new java.awt.Point(70, 30),  // rightEye
                new java.awt.Point(50, 50),  // nose
                new java.awt.Point(35, 70),  // leftMouth
                new java.awt.Point(65, 70)   // rightMouth
            );
        }
    }

    @Nested
    @DisplayName("Equality and HashCode")
    class EqualityAndHashCode {

        @Test
        @DisplayName("Should be equal when same bounds")
        void shouldBeEqualWhenSameBounds() {
            FaceRegion region1 = new FaceRegion(10, 10, 100, 100, 0.9);
            FaceRegion region2 = new FaceRegion(10, 10, 100, 100, 0.5);

            assertThat(region1).isEqualTo(region2);
        }

        @Test
        @DisplayName("Should not be equal when different bounds")
        void shouldNotBeEqualWhenDifferentBounds() {
            FaceRegion region1 = new FaceRegion(10, 10, 100, 100, 0.9);
            FaceRegion region2 = new FaceRegion(20, 10, 100, 100, 0.9);

            assertThat(region1).isNotEqualTo(region2);
        }

        @Test
        @DisplayName("Should have same hashCode for equal regions")
        void shouldHaveSameHashCodeForEqualRegions() {
            FaceRegion region1 = new FaceRegion(10, 10, 100, 100, 0.9);
            FaceRegion region2 = new FaceRegion(10, 10, 100, 100, 0.5);

            assertThat(region1.hashCode()).isEqualTo(region2.hashCode());
        }

        @Test
        @DisplayName("Should not be equal to null")
        void shouldNotBeEqualToNull() {
            FaceRegion region = new FaceRegion(10, 10, 100, 100, 0.9);

            assertThat(region).isNotEqualTo(null);
        }
    }

    @Nested
    @DisplayName("ToString")
    class ToStringTests {

        @Test
        @DisplayName("Should include relevant information")
        void shouldIncludeRelevantInformation() {
            FaceRegion region = new FaceRegion(10, 20, 100, 150, 0.95);

            String str = region.toString();

            assertThat(str).contains("FaceRegion");
            assertThat(str).contains("x=10");
            assertThat(str).contains("y=20");
            assertThat(str).contains("w=100");
            assertThat(str).contains("h=150");
            assertThat(str).contains("conf=0.95");
        }
    }
}
