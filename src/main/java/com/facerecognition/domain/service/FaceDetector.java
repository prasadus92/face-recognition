package com.facerecognition.domain.service;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FaceLandmarks;
import com.facerecognition.domain.model.FaceRegion;

import java.util.List;
import java.util.Optional;

/**
 * Interface for face detection services.
 * Implementations detect face regions and optionally landmarks
 * within input images.
 *
 * <p>Face detection is the first step in the recognition pipeline.
 * It locates faces within an image before feature extraction.</p>
 *
 * <p>Available implementations:</p>
 * <ul>
 *   <li>{@code HaarCascadeDetector} - OpenCV Haar cascades (fast)</li>
 *   <li>{@code HOGDetector} - Histogram of Oriented Gradients</li>
 *   <li>{@code MTCNNDetector} - Multi-task Cascaded CNN (accurate)</li>
 *   <li>{@code RetinaFaceDetector} - State-of-the-art deep learning</li>
 * </ul>
 *
 * @author Face Recognition Team
 * @version 2.0
 * @since 2.0
 * @see FaceRegion
 * @see FaceLandmarks
 */
public interface FaceDetector {

    /**
     * Detects all faces in the given image.
     *
     * @param image the input image
     * @return list of detected face regions, may be empty
     */
    List<FaceRegion> detectFaces(FaceImage image);

    /**
     * Detects all faces in the given image with a confidence threshold.
     *
     * @param image the input image
     * @param minConfidence minimum confidence threshold (0.0 to 1.0)
     * @return list of detected face regions meeting the threshold
     */
    List<FaceRegion> detectFaces(FaceImage image, double minConfidence);

    /**
     * Detects the largest/most prominent face in the image.
     *
     * @param image the input image
     * @return Optional containing the largest face, or empty if none detected
     */
    default Optional<FaceRegion> detectLargestFace(FaceImage image) {
        return detectFaces(image).stream()
            .max((a, b) -> Integer.compare(a.getArea(), b.getArea()));
    }

    /**
     * Detects the face with highest confidence in the image.
     *
     * @param image the input image
     * @return Optional containing the most confident detection, or empty
     */
    default Optional<FaceRegion> detectMostConfidentFace(FaceImage image) {
        return detectFaces(image).stream()
            .max((a, b) -> Double.compare(a.getConfidence(), b.getConfidence()));
    }

    /**
     * Checks if the detector supports landmark detection.
     *
     * @return true if landmarks can be detected
     */
    boolean supportsLandmarks();

    /**
     * Detects facial landmarks for a given face region.
     *
     * @param image the full image
     * @param faceRegion the detected face region
     * @return Optional containing landmarks, or empty if not supported/failed
     */
    Optional<FaceLandmarks> detectLandmarks(FaceImage image, FaceRegion faceRegion);

    /**
     * Gets the name of this detector implementation.
     *
     * @return the detector name
     */
    String getName();

    /**
     * Gets the version of this detector.
     *
     * @return the version string
     */
    String getVersion();

    /**
     * Gets the minimum face size this detector can reliably detect.
     *
     * @return minimum face size in pixels
     */
    int getMinFaceSize();

    /**
     * Sets the minimum face size to detect.
     *
     * @param minSize the minimum size in pixels
     */
    void setMinFaceSize(int minSize);

    /**
     * Configuration options for face detection.
     */
    class DetectionConfig {
        private double minConfidence = 0.5;
        private int minFaceSize = 30;
        private boolean detectLandmarks = true;
        private int maxFaces = 10;
        private double scaleFactor = 1.1;

        public double getMinConfidence() { return minConfidence; }
        public DetectionConfig setMinConfidence(double c) { this.minConfidence = c; return this; }

        public int getMinFaceSize() { return minFaceSize; }
        public DetectionConfig setMinFaceSize(int s) { this.minFaceSize = s; return this; }

        public boolean isDetectLandmarks() { return detectLandmarks; }
        public DetectionConfig setDetectLandmarks(boolean d) { this.detectLandmarks = d; return this; }

        public int getMaxFaces() { return maxFaces; }
        public DetectionConfig setMaxFaces(int m) { this.maxFaces = m; return this; }

        public double getScaleFactor() { return scaleFactor; }
        public DetectionConfig setScaleFactor(double s) { this.scaleFactor = s; return this; }
    }
}
