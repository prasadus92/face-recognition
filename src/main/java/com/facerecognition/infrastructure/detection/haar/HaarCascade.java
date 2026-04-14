package com.facerecognition.infrastructure.detection.haar;

import java.io.Serializable;
import java.util.Arrays;

/**
 * In-memory representation of a trained OpenCV Haar cascade classifier.
 *
 * <p>A cascade is an ordered list of <em>stages</em>. Each stage contains a set
 * of <em>weak classifiers</em> (decision stumps) that each reference one of the
 * cascade's <em>Haar-like features</em> by index. During detection, a candidate
 * window is rejected as soon as any stage's summed weak-classifier response
 * falls below that stage's threshold. A window is accepted as a face only if
 * it survives every stage.</p>
 *
 * <p>This class is an immutable data holder; it does not know how to evaluate
 * itself. The evaluation loop lives in
 * {@link com.facerecognition.infrastructure.detection.HaarCascadeFaceDetector}
 * so that the data model stays allocation-free and cacheable.</p>
 *
 * <p>The companion {@link HaarCascadeLoader} parses OpenCV's
 * {@code haarcascade_*.xml} files into instances of this class.</p>
 *
 * @see HaarCascadeLoader
 * @see com.facerecognition.infrastructure.detection.HaarCascadeFaceDetector
 */
public final class HaarCascade implements Serializable {

    private static final long serialVersionUID = 1L;

    private final int windowWidth;
    private final int windowHeight;
    private final Stage[] stages;
    private final Feature[] features;

    public HaarCascade(int windowWidth, int windowHeight, Stage[] stages, Feature[] features) {
        if (windowWidth <= 0 || windowHeight <= 0) {
            throw new IllegalArgumentException("Window dimensions must be positive");
        }
        if (stages == null || stages.length == 0) {
            throw new IllegalArgumentException("Cascade must have at least one stage");
        }
        if (features == null || features.length == 0) {
            throw new IllegalArgumentException("Cascade must have at least one feature");
        }
        this.windowWidth = windowWidth;
        this.windowHeight = windowHeight;
        this.stages = stages.clone();
        this.features = features.clone();
    }

    /** @return the trained detector window width (typically 24 for frontal faces). */
    public int getWindowWidth() {
        return windowWidth;
    }

    /** @return the trained detector window height (typically 24 for frontal faces). */
    public int getWindowHeight() {
        return windowHeight;
    }

    /** @return the cascade stages, in evaluation order. Do not mutate. */
    public Stage[] getStages() {
        return stages;
    }

    /** @return the cascade's feature pool, indexed by {@link WeakClassifier#getFeatureIndex()}. Do not mutate. */
    public Feature[] getFeatures() {
        return features;
    }

    /** @return number of stages in the cascade. */
    public int getStageCount() {
        return stages.length;
    }

    /** @return number of features in the cascade's feature pool. */
    public int getFeatureCount() {
        return features.length;
    }

    @Override
    public String toString() {
        return "HaarCascade{window=" + windowWidth + "x" + windowHeight
                + ", stages=" + stages.length + ", features=" + features.length + "}";
    }

    // ---------------------------------------------------------------------
    // Nested data classes
    // ---------------------------------------------------------------------

    /**
     * One stage of the cascade. Evaluation sums the contributions from every
     * weak classifier and rejects the window if the sum falls below
     * {@link #getThreshold()}.
     */
    public static final class Stage implements Serializable {
        private static final long serialVersionUID = 1L;

        private final float threshold;
        private final WeakClassifier[] classifiers;

        public Stage(float threshold, WeakClassifier[] classifiers) {
            this.threshold = threshold;
            this.classifiers = classifiers.clone();
        }

        public float getThreshold() {
            return threshold;
        }

        public WeakClassifier[] getClassifiers() {
            return classifiers;
        }
    }

    /**
     * A single weak classifier: a decision stump that evaluates one Haar feature
     * and chooses between a "below threshold" leaf value and an "above threshold"
     * leaf value.
     */
    public static final class WeakClassifier implements Serializable {
        private static final long serialVersionUID = 1L;

        private final int featureIndex;
        private final float nodeThreshold;
        private final float leftLeaf;
        private final float rightLeaf;

        public WeakClassifier(int featureIndex, float nodeThreshold, float leftLeaf, float rightLeaf) {
            this.featureIndex = featureIndex;
            this.nodeThreshold = nodeThreshold;
            this.leftLeaf = leftLeaf;
            this.rightLeaf = rightLeaf;
        }

        public int getFeatureIndex() {
            return featureIndex;
        }

        public float getNodeThreshold() {
            return nodeThreshold;
        }

        public float getLeftLeaf() {
            return leftLeaf;
        }

        public float getRightLeaf() {
            return rightLeaf;
        }
    }

    /**
     * A Haar-like feature composed of 2 or 3 weighted rectangles. Each
     * rectangle's coordinates are expressed in the trained window's coordinate
     * system ({@link HaarCascade#getWindowWidth()} × {@link HaarCascade#getWindowHeight()})
     * and must be scaled to the current sliding-window size at detection time.
     */
    public static final class Feature implements Serializable {
        private static final long serialVersionUID = 1L;

        private final Rect[] rects;

        public Feature(Rect[] rects) {
            if (rects == null || rects.length < 2 || rects.length > 3) {
                throw new IllegalArgumentException(
                        "Haar feature must have 2 or 3 rectangles, got "
                                + (rects == null ? "null" : String.valueOf(rects.length)));
            }
            this.rects = rects.clone();
        }

        public Rect[] getRects() {
            return rects;
        }

        @Override
        public String toString() {
            return "Feature" + Arrays.toString(rects);
        }
    }

    /** A weighted rectangle inside a Haar feature. */
    public static final class Rect implements Serializable {
        private static final long serialVersionUID = 1L;

        private final int x;
        private final int y;
        private final int width;
        private final int height;
        private final float weight;

        public Rect(int x, int y, int width, int height, float weight) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
            this.weight = weight;
        }

        public int getX() {
            return x;
        }

        public int getY() {
            return y;
        }

        public int getWidth() {
            return width;
        }

        public int getHeight() {
            return height;
        }

        public float getWeight() {
            return weight;
        }

        @Override
        public String toString() {
            return "(" + x + "," + y + "," + width + "," + height + "," + weight + ")";
        }
    }
}
