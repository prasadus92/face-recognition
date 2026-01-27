package com.facerecognition.domain.model;

import java.awt.Rectangle;
import java.io.Serializable;
import java.util.Objects;

/**
 * Represents a detected face region within an image.
 * Contains bounding box coordinates and detection confidence.
 *
 * <p>FaceRegion is used to mark the location of a detected face
 * within a larger image. It provides methods for calculating
 * overlap, containment, and extracting the face subimage.</p>
 *
 * @author Face Recognition Team
 * @version 2.0
 * @since 2.0
 * @see FaceImage
 * @see FaceLandmarks
 */
public class FaceRegion implements Serializable {

    private static final long serialVersionUID = 1L;

    private final int x;
    private final int y;
    private final int width;
    private final int height;
    private final double confidence;
    private final FaceLandmarks landmarks;

    /**
     * Creates a new FaceRegion with specified bounds and confidence.
     *
     * @param x the x-coordinate of the top-left corner
     * @param y the y-coordinate of the top-left corner
     * @param width the width of the face region
     * @param height the height of the face region
     * @param confidence the detection confidence (0.0 to 1.0)
     * @throws IllegalArgumentException if dimensions are negative or confidence is out of range
     */
    public FaceRegion(int x, int y, int width, int height, double confidence) {
        this(x, y, width, height, confidence, null);
    }

    /**
     * Creates a new FaceRegion with bounds, confidence, and landmarks.
     *
     * @param x the x-coordinate of the top-left corner
     * @param y the y-coordinate of the top-left corner
     * @param width the width of the face region
     * @param height the height of the face region
     * @param confidence the detection confidence (0.0 to 1.0)
     * @param landmarks the detected facial landmarks (may be null)
     */
    public FaceRegion(int x, int y, int width, int height, double confidence, FaceLandmarks landmarks) {
        if (width <= 0 || height <= 0) {
            throw new IllegalArgumentException("Width and height must be positive");
        }
        if (confidence < 0.0 || confidence > 1.0) {
            throw new IllegalArgumentException("Confidence must be between 0.0 and 1.0");
        }

        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.confidence = confidence;
        this.landmarks = landmarks;
    }

    /**
     * Creates a FaceRegion from a Rectangle with specified confidence.
     *
     * @param rect the bounding rectangle
     * @param confidence the detection confidence
     * @return a new FaceRegion
     */
    public static FaceRegion fromRectangle(Rectangle rect, double confidence) {
        return new FaceRegion(rect.x, rect.y, rect.width, rect.height, confidence);
    }

    /**
     * Gets the x-coordinate of the top-left corner.
     *
     * @return the x-coordinate
     */
    public int getX() {
        return x;
    }

    /**
     * Gets the y-coordinate of the top-left corner.
     *
     * @return the y-coordinate
     */
    public int getY() {
        return y;
    }

    /**
     * Gets the width of the face region.
     *
     * @return the width in pixels
     */
    public int getWidth() {
        return width;
    }

    /**
     * Gets the height of the face region.
     *
     * @return the height in pixels
     */
    public int getHeight() {
        return height;
    }

    /**
     * Gets the detection confidence score.
     *
     * @return confidence between 0.0 and 1.0
     */
    public double getConfidence() {
        return confidence;
    }

    /**
     * Gets the facial landmarks if available.
     *
     * @return the landmarks, or null if not detected
     */
    public FaceLandmarks getLandmarks() {
        return landmarks;
    }

    /**
     * Checks if landmarks are available for this face.
     *
     * @return true if landmarks were detected
     */
    public boolean hasLandmarks() {
        return landmarks != null;
    }

    /**
     * Gets the center x-coordinate of the face region.
     *
     * @return the center x-coordinate
     */
    public int getCenterX() {
        return x + width / 2;
    }

    /**
     * Gets the center y-coordinate of the face region.
     *
     * @return the center y-coordinate
     */
    public int getCenterY() {
        return y + height / 2;
    }

    /**
     * Gets the area of the face region in pixels.
     *
     * @return the area in square pixels
     */
    public int getArea() {
        return width * height;
    }

    /**
     * Gets the aspect ratio (width/height).
     *
     * @return the aspect ratio
     */
    public double getAspectRatio() {
        return (double) width / height;
    }

    /**
     * Converts to an AWT Rectangle.
     *
     * @return a Rectangle with the same bounds
     */
    public Rectangle toRectangle() {
        return new Rectangle(x, y, width, height);
    }

    /**
     * Checks if this region contains a point.
     *
     * @param px the x-coordinate of the point
     * @param py the y-coordinate of the point
     * @return true if the point is within this region
     */
    public boolean contains(int px, int py) {
        return px >= x && px < x + width && py >= y && py < y + height;
    }

    /**
     * Checks if this region contains another region completely.
     *
     * @param other the other region
     * @return true if other is completely inside this region
     */
    public boolean contains(FaceRegion other) {
        return x <= other.x && y <= other.y &&
               x + width >= other.x + other.width &&
               y + height >= other.y + other.height;
    }

    /**
     * Calculates the Intersection over Union (IoU) with another region.
     * IoU is a common metric for evaluating detection accuracy.
     *
     * @param other the other face region
     * @return IoU value between 0.0 (no overlap) and 1.0 (identical)
     */
    public double intersectionOverUnion(FaceRegion other) {
        int x1 = Math.max(this.x, other.x);
        int y1 = Math.max(this.y, other.y);
        int x2 = Math.min(this.x + this.width, other.x + other.width);
        int y2 = Math.min(this.y + this.height, other.y + other.height);

        if (x1 >= x2 || y1 >= y2) {
            return 0.0; // No intersection
        }

        int intersectionArea = (x2 - x1) * (y2 - y1);
        int unionArea = this.getArea() + other.getArea() - intersectionArea;

        return (double) intersectionArea / unionArea;
    }

    /**
     * Creates an expanded version of this region.
     *
     * @param factor the expansion factor (1.0 = no change, 1.5 = 50% larger)
     * @return a new FaceRegion with expanded bounds
     */
    public FaceRegion expand(double factor) {
        int newWidth = (int) (width * factor);
        int newHeight = (int) (height * factor);
        int newX = x - (newWidth - width) / 2;
        int newY = y - (newHeight - height) / 2;
        return new FaceRegion(newX, newY, newWidth, newHeight, confidence, landmarks);
    }

    /**
     * Creates a FaceRegion with new landmarks.
     *
     * @param newLandmarks the landmarks to attach
     * @return a new FaceRegion with landmarks
     */
    public FaceRegion withLandmarks(FaceLandmarks newLandmarks) {
        return new FaceRegion(x, y, width, height, confidence, newLandmarks);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        FaceRegion that = (FaceRegion) o;
        return x == that.x && y == that.y && width == that.width && height == that.height;
    }

    @Override
    public int hashCode() {
        return Objects.hash(x, y, width, height);
    }

    @Override
    public String toString() {
        return String.format("FaceRegion{x=%d, y=%d, w=%d, h=%d, conf=%.2f}",
            x, y, width, height, confidence);
    }
}
