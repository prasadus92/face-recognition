package com.facerecognition.infrastructure.detection;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Optional;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FaceLandmarks;
import com.facerecognition.domain.model.FaceRegion;
import com.facerecognition.domain.service.FaceDetector;
import com.facerecognition.infrastructure.detection.haar.HaarCascade;
import com.facerecognition.infrastructure.detection.haar.HaarCascadeLoader;

/**
 * Pure-Java Viola-Jones face detector that runs OpenCV's trained Haar cascades.
 *
 * <p>It follows {@code cv::CascadeClassifier::detectMultiScale}: the grayscale image is resized
 * into a pyramid (one level per {@code scaleFactor} step, starting where the cascade's window
 * equals {@code minFaceSize}), the cascade's native window slides over each level with its
 * original rectangles, every weak classifier compares its weighted rectangle sum with its
 * threshold times the window's standard-deviation term, and hits are grouped with OpenCV's
 * {@code groupRectangles} (see {@link RectangleGrouping}): a face needs more than
 * {@code minNeighbours} supporting hits. The finest level is at most {@value #MAX_SCAN_SIDE}
 * pixels on its longest side, which bounds the work per image; in larger photos the smallest
 * detectable face grows proportionally.</p>
 *
 * <p>Instances are safe for concurrent use: the cascade is immutable and each call allocates its
 * own buffers.</p>
 */
public final class HaarCascadeFaceDetector implements FaceDetector, Serializable {

    private static final long serialVersionUID = 1L;
    private static final Logger LOG = LoggerFactory.getLogger(HaarCascadeFaceDetector.class);

    /** Algorithm name reported to persistence and metrics. */
    public static final String ALGORITHM_NAME = "HaarCascade";

    /** Classpath location of the default frontal-face cascade bundled with the library. */
    public static final String DEFAULT_CASCADE_RESOURCE = "cascades/haarcascade_frontalface_default.xml";

    /** Default scale step between successive pyramid levels. */
    public static final double DEFAULT_SCALE_FACTOR = 1.1;

    /** Default minimum neighbour count required for a candidate to survive NMS. */
    public static final int DEFAULT_MIN_NEIGHBOURS = 3;

    /** Longest side of the finest pyramid level, in pixels. */
    public static final int MAX_SCAN_SIDE = 1024;

    /** Default minimum detection size, matching OpenCV's typical setup. */
    public static final int DEFAULT_MIN_FACE_SIZE = 24;

    private final HaarCascade cascade;
    private volatile int minFaceSize;
    private final double scaleFactor;
    private final int minNeighbours;

    // ---------------------------------------------------------------------
    // Construction
    // ---------------------------------------------------------------------

    /** Creates a detector using the bundled frontal-face cascade and default parameters. */
    public HaarCascadeFaceDetector() {
        this(loadDefaultCascade(), DEFAULT_MIN_FACE_SIZE, DEFAULT_SCALE_FACTOR, DEFAULT_MIN_NEIGHBOURS);
    }

    /** Creates a detector from a caller-supplied cascade with default detection parameters. */
    public HaarCascadeFaceDetector(HaarCascade cascade) {
        this(cascade, DEFAULT_MIN_FACE_SIZE, DEFAULT_SCALE_FACTOR, DEFAULT_MIN_NEIGHBOURS);
    }

    /**
     * Full constructor.
     *
     * @param cascade       the parsed cascade (never {@code null})
     * @param minFaceSize   minimum detection window side in pixels; smaller windows are skipped
     * @param scaleFactor   scale step between successive pyramid levels (must be &gt; 1.0)
     * @param minNeighbours a face needs more than this many supporting raw hits, as in OpenCV
     */
    public HaarCascadeFaceDetector(HaarCascade cascade, int minFaceSize, double scaleFactor, int minNeighbours) {
        if (cascade == null) {
            throw new IllegalArgumentException("cascade must not be null");
        }
        if (minFaceSize < cascade.getWindowWidth()) {
            // Detections below the trained window size are impossible; silently
            // raise the floor to the window size so callers don't get confused.
            minFaceSize = cascade.getWindowWidth();
        }
        if (scaleFactor <= 1.0) {
            throw new IllegalArgumentException("scaleFactor must be > 1.0, was " + scaleFactor);
        }
        if (minNeighbours < 1) {
            throw new IllegalArgumentException("minNeighbours must be >= 1, was " + minNeighbours);
        }
        this.cascade = cascade;
        this.minFaceSize = minFaceSize;
        this.scaleFactor = scaleFactor;
        this.minNeighbours = minNeighbours;
        LOG.debug("HaarCascadeFaceDetector initialized: {}, minFaceSize={}, scaleFactor={}, minNeighbours={}",
                cascade, minFaceSize, scaleFactor, minNeighbours);
    }

    /**
     * Loads the bundled OpenCV frontal-face cascade from the classpath. Exposed so
     * that configuration / auto-wiring code can share one parsed cascade instance
     * across detectors instead of each detector re-parsing the 900 KB XML.
     */
    public static HaarCascade loadDefaultCascade() {
        try {
            return HaarCascadeLoader.loadFromClasspath(DEFAULT_CASCADE_RESOURCE);
        } catch (IOException e) {
            throw new IllegalStateException(
                    "Failed to load bundled Haar cascade from classpath: " + DEFAULT_CASCADE_RESOURCE, e);
        }
    }

    // ---------------------------------------------------------------------
    // FaceDetector surface
    // ---------------------------------------------------------------------

    @Override
    public List<FaceRegion> detectFaces(FaceImage image) {
        return detectFaces(image, 0.0);
    }

    @Override
    public List<FaceRegion> detectFaces(FaceImage image, double minConfidence) {
        if (image == null) {
            return Collections.emptyList();
        }
        BufferedImage src = image.getImage();
        int width = src.getWidth();
        int height = src.getHeight();
        int windowW = cascade.getWindowWidth();
        int windowH = cascade.getWindowHeight();
        int[] grayscale = toGrayscale(src, width, height);

        List<RectangleGrouping.Box> raw = new ArrayList<>();
        double firstFactor = Math.max((double) minFaceSize / windowW,
                (double) Math.max(width, height) / MAX_SCAN_SIDE);
        for (double factor = firstFactor; ; factor *= scaleFactor) {
            int scaledW = (int) Math.round(width / factor);
            int scaledH = (int) Math.round(height / factor);
            if (scaledW < windowW || scaledH < windowH) {
                break;
            }
            int[] scaled = resizeBilinear(grayscale, width, height, scaledW, scaledH);
            long[] integral = new long[(scaledW + 1) * (scaledH + 1)];
            long[] squaredIntegral = new long[(scaledW + 1) * (scaledH + 1)];
            buildIntegralImages(scaled, scaledW, scaledH, integral, squaredIntegral);
            // OpenCV steps 2 pixels on fine levels and 1 pixel once the image is downscaled 2x or more.
            int step = factor >= 2.0 ? 1 : 2;
            scanLevel(integral, squaredIntegral, scaledW, scaledH, step, factor, raw);
        }

        List<FaceRegion> regions = mergeDetections(raw, minConfidence);
        LOG.debug("Haar detection: {} raw candidates -> {} regions", raw.size(), regions.size());
        return regions;
    }

    @Override
    public boolean supportsLandmarks() {
        return false;
    }

    @Override
    public Optional<FaceLandmarks> detectLandmarks(FaceImage image, FaceRegion faceRegion) {
        return Optional.empty();
    }

    @Override
    public String getName() {
        return ALGORITHM_NAME;
    }

    @Override
    public String getVersion() {
        return "1.0";
    }

    @Override
    public int getMinFaceSize() {
        return minFaceSize;
    }

    @Override
    public void setMinFaceSize(int minSize) {
        this.minFaceSize = Math.max(minSize, cascade.getWindowWidth());
    }

    // ---------------------------------------------------------------------
    // Cascade evaluation
    // ---------------------------------------------------------------------

    /** Slides the cascade's native window over one pyramid level, mapping hits back to source pixels. */
    private void scanLevel(long[] integral,
                           long[] squaredIntegral,
                           int levelW,
                           int levelH,
                           int step,
                           double factor,
                           List<RectangleGrouping.Box> out) {
        int stride = levelW + 1;
        int windowW = cascade.getWindowWidth();
        int windowH = cascade.getWindowHeight();
        int detectedW = (int) Math.round(windowW * factor);
        int detectedH = (int) Math.round(windowH * factor);
        LevelFeatures features = new LevelFeatures(cascade.getFeatures(), stride);
        for (int y = 0; y + windowH <= levelH; y += step) {
            for (int x = 0; x + windowW <= levelW; x += step) {
                if (passesCascade(integral, squaredIntegral, stride, x, y, features)) {
                    out.add(new RectangleGrouping.Box((int) Math.round(x * factor), (int) Math.round(y * factor),
                            detectedW, detectedH));
                }
            }
        }
    }

    /**
     * Evaluates every stage at one window position. As in OpenCV, a weak classifier compares its
     * weighted rectangle sum with its threshold times the standard-deviation term of the window
     * shrunk by 1 pixel on each side.
     */
    private boolean passesCascade(long[] integral, long[] squaredIntegral, int stride, int x, int y,
                                  LevelFeatures features) {
        int innerW = cascade.getWindowWidth() - 2;
        int innerH = cascade.getWindowHeight() - 2;
        double area = (double) innerW * innerH;
        double sum = rectSum(integral, stride, x + 1, y + 1, innerW, innerH);
        double squaredSum = rectSum(squaredIntegral, stride, x + 1, y + 1, innerW, innerH);
        double spread = area * squaredSum - sum * sum;
        double norm = spread > 0 ? Math.sqrt(spread) : 1.0;

        int origin = y * stride + x;
        for (HaarCascade.Stage stage : cascade.getStages()) {
            double stageSum = 0.0;
            for (HaarCascade.WeakClassifier weak : stage.getClassifiers()) {
                double value = features.value(integral, origin, weak.getFeatureIndex());
                stageSum += value < weak.getNodeThreshold() * norm ? weak.getLeftLeaf() : weak.getRightLeaf();
            }
            if (stageSum < stage.getThreshold()) {
                return false;
            }
        }
        return true;
    }

    /** Rectangle corners of every feature as offsets into one level's integral image, plus weights. */
    private static final class LevelFeatures {
        private final int[][] corners;
        private final double[][] weights;

        LevelFeatures(HaarCascade.Feature[] features, int stride) {
            corners = new int[features.length][];
            weights = new double[features.length][];
            for (int f = 0; f < features.length; f++) {
                HaarCascade.Rect[] rects = features[f].getRects();
                corners[f] = new int[rects.length * 4];
                weights[f] = new double[rects.length];
                for (int r = 0; r < rects.length; r++) {
                    HaarCascade.Rect rect = rects[r];
                    int topLeft = rect.getY() * stride + rect.getX();
                    int bottomLeft = topLeft + rect.getHeight() * stride;
                    corners[f][r * 4] = topLeft;
                    corners[f][r * 4 + 1] = topLeft + rect.getWidth();
                    corners[f][r * 4 + 2] = bottomLeft;
                    corners[f][r * 4 + 3] = bottomLeft + rect.getWidth();
                    weights[f][r] = rect.getWeight();
                }
            }
        }

        double value(long[] integral, int origin, int feature) {
            int[] c = corners[feature];
            double[] w = weights[feature];
            double value = 0.0;
            for (int r = 0; r < w.length; r++) {
                value += w[r] * (integral[c[r * 4 + 3] + origin] - integral[c[r * 4 + 1] + origin]
                        - integral[c[r * 4 + 2] + origin] + integral[c[r * 4] + origin]);
            }
            return value;
        }
    }

    /** Bilinear resize with pixel-center alignment, as {@code cv::resize} with {@code INTER_LINEAR}. */
    static int[] resizeBilinear(int[] src, int srcW, int srcH, int dstW, int dstH) {
        int[] dst = new int[dstW * dstH];
        double scaleX = (double) srcW / dstW;
        double scaleY = (double) srcH / dstH;
        for (int dy = 0; dy < dstH; dy++) {
            double sy = Math.max(0.0, (dy + 0.5) * scaleY - 0.5);
            int y0 = Math.min((int) sy, srcH - 1);
            int y1 = Math.min(y0 + 1, srcH - 1);
            double fy = sy - y0;
            for (int dx = 0; dx < dstW; dx++) {
                double sx = Math.max(0.0, (dx + 0.5) * scaleX - 0.5);
                int x0 = Math.min((int) sx, srcW - 1);
                int x1 = Math.min(x0 + 1, srcW - 1);
                double fx = sx - x0;
                double top = src[y0 * srcW + x0] * (1 - fx) + src[y0 * srcW + x1] * fx;
                double bottom = src[y1 * srcW + x0] * (1 - fx) + src[y1 * srcW + x1] * fx;
                dst[dy * dstW + dx] = (int) Math.round(top * (1 - fy) + bottom * fy);
            }
        }
        return dst;
    }

    private static long rectSum(long[] integral, int stride, int x, int y, int w, int h) {
        int tl = y * stride + x;
        int tr = tl + w;
        int bl = tl + h * stride;
        int br = bl + w;
        return integral[br] - integral[tr] - integral[bl] + integral[tl];
    }

    /** Builds integral + squared integral image in one pass. */
    private static void buildIntegralImages(int[] grayscale,
                                            int width,
                                            int height,
                                            long[] integral,
                                            long[] squaredIntegral) {
        int stride = width + 1;
        for (int y = 0; y < height; y++) {
            long rowSum = 0;
            long rowSumSq = 0;
            int srcRow = y * width;
            int dstRow = (y + 1) * stride;
            int prevRow = y * stride;
            for (int x = 0; x < width; x++) {
                int px = grayscale[srcRow + x];
                rowSum += px;
                rowSumSq += (long) px * px;
                integral[dstRow + x + 1] = integral[prevRow + x + 1] + rowSum;
                squaredIntegral[dstRow + x + 1] = squaredIntegral[prevRow + x + 1] + rowSumSq;
            }
        }
    }

    /** Extracts a grayscale {@code int[]} from a {@link BufferedImage}. */
    private static int[] toGrayscale(BufferedImage src, int width, int height) {
        int[] rgb = new int[width * height];
        src.getRGB(0, 0, width, height, rgb, 0, width);
        int[] gray = new int[width * height];
        for (int i = 0; i < rgb.length; i++) {
            int p = rgb[i];
            int r = (p >> 16) & 0xff;
            int g = (p >> 8) & 0xff;
            int b = p & 0xff;
            // Standard ITU BT.601 luma.
            gray[i] = (r * 299 + g * 587 + b * 114) / 1000;
        }
        return gray;
    }

    // ---------------------------------------------------------------------
    // Non-maximum suppression
    // ---------------------------------------------------------------------

    /**
     * Groups raw hits as OpenCV does. Confidence is a group's support relative to the best-supported
     * group in the same image, so it ranks faces within one image but is not comparable across images.
     */
    private List<FaceRegion> mergeDetections(List<RectangleGrouping.Box> raw, double minConfidence) {
        List<RectangleGrouping.Group> groups = RectangleGrouping.group(raw, minNeighbours);
        int maxSupport = groups.stream().mapToInt(RectangleGrouping.Group::neighbours).max().orElse(1);
        List<FaceRegion> regions = new ArrayList<>();
        for (RectangleGrouping.Group group : groups) {
            double confidence = (double) group.neighbours() / maxSupport;
            if (confidence >= minConfidence) {
                RectangleGrouping.Box box = group.box();
                regions.add(new FaceRegion(box.x(), box.y(), box.width(), box.height(), confidence));
            }
        }
        return regions;
    }
}
