package com.facerecognition.domain.model;

import java.awt.Point;
import java.io.Serializable;
import java.util.Arrays;
import java.util.Objects;

/**
 * Represents facial landmarks (keypoints) detected on a face.
 * Landmarks are used for face alignment before feature extraction.
 *
 * <p>This class supports various landmark configurations:</p>
 * <ul>
 *   <li>5-point: eyes (2), nose (1), mouth corners (2)</li>
 *   <li>68-point: standard dlib configuration</li>
 *   <li>106-point: detailed landmark set</li>
 * </ul>
 *
 * <p>Example usage:</p>
 * <pre>{@code
 * FaceLandmarks landmarks = FaceLandmarks.create5Point(
 *     new Point(50, 80),   // left eye
 *     new Point(110, 80),  // right eye
 *     new Point(80, 120),  // nose tip
 *     new Point(55, 150),  // left mouth
 *     new Point(105, 150)  // right mouth
 * );
 * double angle = landmarks.getFaceRotationAngle();
 * }</pre>
 *
 * @author Face Recognition Team
 * @version 2.0
 * @since 2.0
 */
public class FaceLandmarks implements Serializable {

    private static final long serialVersionUID = 1L;

    /** Index constants for 5-point landmarks. */
    public static final int LEFT_EYE = 0;
    public static final int RIGHT_EYE = 1;
    public static final int NOSE_TIP = 2;
    public static final int LEFT_MOUTH = 3;
    public static final int RIGHT_MOUTH = 4;

    private final Point[] points;
    private final LandmarkType type;
    private final double detectionConfidence;

    /**
     * Types of landmark configurations.
     */
    public enum LandmarkType {
        /** 5 key points: eyes, nose, mouth corners. */
        FIVE_POINT(5),
        /** 68 points: dlib standard configuration. */
        SIXTY_EIGHT_POINT(68),
        /** 106 points: detailed landmarks. */
        DETAILED(106);

        private final int pointCount;

        LandmarkType(int pointCount) {
            this.pointCount = pointCount;
        }

        public int getPointCount() {
            return pointCount;
        }
    }

    /**
     * Creates landmarks with the specified points and type.
     *
     * @param points the landmark points
     * @param type the landmark type
     * @param confidence the detection confidence
     */
    private FaceLandmarks(Point[] points, LandmarkType type, double confidence) {
        if (points.length != type.getPointCount()) {
            throw new IllegalArgumentException(
                String.format("Expected %d points for %s, got %d",
                    type.getPointCount(), type, points.length));
        }
        this.points = Arrays.copyOf(points, points.length);
        this.type = type;
        this.detectionConfidence = confidence;
    }

    /**
     * Creates 5-point landmarks from individual points.
     *
     * @param leftEye center of left eye
     * @param rightEye center of right eye
     * @param noseTip tip of nose
     * @param leftMouth left corner of mouth
     * @param rightMouth right corner of mouth
     * @return a new FaceLandmarks instance
     */
    public static FaceLandmarks create5Point(
            Point leftEye, Point rightEye, Point noseTip,
            Point leftMouth, Point rightMouth) {
        return create5Point(leftEye, rightEye, noseTip, leftMouth, rightMouth, 1.0);
    }

    /**
     * Creates 5-point landmarks with confidence score.
     *
     * @param leftEye center of left eye
     * @param rightEye center of right eye
     * @param noseTip tip of nose
     * @param leftMouth left corner of mouth
     * @param rightMouth right corner of mouth
     * @param confidence detection confidence
     * @return a new FaceLandmarks instance
     */
    public static FaceLandmarks create5Point(
            Point leftEye, Point rightEye, Point noseTip,
            Point leftMouth, Point rightMouth, double confidence) {
        Point[] points = new Point[] {
            new Point(leftEye),
            new Point(rightEye),
            new Point(noseTip),
            new Point(leftMouth),
            new Point(rightMouth)
        };
        return new FaceLandmarks(points, LandmarkType.FIVE_POINT, confidence);
    }

    /**
     * Creates landmarks from a point array.
     *
     * @param points the landmark points
     * @param type the landmark type
     * @param confidence the detection confidence
     * @return a new FaceLandmarks instance
     */
    public static FaceLandmarks create(Point[] points, LandmarkType type, double confidence) {
        return new FaceLandmarks(points, type, confidence);
    }

    /**
     * Gets the landmark type.
     *
     * @return the LandmarkType
     */
    public LandmarkType getType() {
        return type;
    }

    /**
     * Gets the number of landmark points.
     *
     * @return the point count
     */
    public int getPointCount() {
        return points.length;
    }

    /**
     * Gets a specific landmark point by index.
     *
     * @param index the point index
     * @return the Point at the specified index
     * @throws IndexOutOfBoundsException if index is invalid
     */
    public Point getPoint(int index) {
        return new Point(points[index]);
    }

    /**
     * Gets all landmark points.
     *
     * @return a copy of the points array
     */
    public Point[] getAllPoints() {
        return Arrays.copyOf(points, points.length);
    }

    /**
     * Gets the detection confidence.
     *
     * @return confidence between 0.0 and 1.0
     */
    public double getConfidence() {
        return detectionConfidence;
    }

    /**
     * Gets the left eye center (for 5-point landmarks).
     *
     * @return the left eye point
     */
    public Point getLeftEye() {
        return getPoint(LEFT_EYE);
    }

    /**
     * Gets the right eye center (for 5-point landmarks).
     *
     * @return the right eye point
     */
    public Point getRightEye() {
        return getPoint(RIGHT_EYE);
    }

    /**
     * Gets the nose tip point (for 5-point landmarks).
     *
     * @return the nose tip point
     */
    public Point getNoseTip() {
        return getPoint(NOSE_TIP);
    }

    /**
     * Calculates the eye center point (midpoint between eyes).
     *
     * @return the eye center point
     */
    public Point getEyeCenter() {
        Point left = getLeftEye();
        Point right = getRightEye();
        return new Point((left.x + right.x) / 2, (left.y + right.y) / 2);
    }

    /**
     * Calculates the interocular distance (distance between eyes).
     *
     * @return the distance in pixels
     */
    public double getInterocularDistance() {
        Point left = getLeftEye();
        Point right = getRightEye();
        return Math.sqrt(Math.pow(right.x - left.x, 2) + Math.pow(right.y - left.y, 2));
    }

    /**
     * Calculates the face rotation angle based on eye positions.
     * Positive values indicate clockwise rotation.
     *
     * @return the rotation angle in degrees
     */
    public double getFaceRotationAngle() {
        Point left = getLeftEye();
        Point right = getRightEye();
        double deltaY = right.y - left.y;
        double deltaX = right.x - left.x;
        return Math.toDegrees(Math.atan2(deltaY, deltaX));
    }

    /**
     * Estimates the face scale based on interocular distance.
     * Assumes standard adult interocular distance of ~63mm.
     *
     * @param standardInterocular the expected interocular distance in pixels
     * @return the scale factor relative to standard size
     */
    public double getFaceScale(double standardInterocular) {
        return getInterocularDistance() / standardInterocular;
    }

    /**
     * Calculates the mouth width.
     *
     * @return the mouth width in pixels
     */
    public double getMouthWidth() {
        Point left = getPoint(LEFT_MOUTH);
        Point right = getPoint(RIGHT_MOUTH);
        return Math.sqrt(Math.pow(right.x - left.x, 2) + Math.pow(right.y - left.y, 2));
    }

    /**
     * Gets the mouth center point.
     *
     * @return the mouth center point
     */
    public Point getMouthCenter() {
        Point left = getPoint(LEFT_MOUTH);
        Point right = getPoint(RIGHT_MOUTH);
        return new Point((left.x + right.x) / 2, (left.y + right.y) / 2);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        FaceLandmarks that = (FaceLandmarks) o;
        return type == that.type && Arrays.equals(points, that.points);
    }

    @Override
    public int hashCode() {
        int result = Objects.hash(type);
        result = 31 * result + Arrays.hashCode(points);
        return result;
    }

    @Override
    public String toString() {
        return String.format("FaceLandmarks{type=%s, points=%d, rotation=%.1f°}",
            type, points.length, getFaceRotationAngle());
    }
}
