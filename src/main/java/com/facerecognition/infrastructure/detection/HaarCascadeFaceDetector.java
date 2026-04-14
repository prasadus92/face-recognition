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
 * Pure-Java Viola-Jones face detector using real Haar cascades from OpenCV.
 *
 * <p>Loads a {@link HaarCascade} (typically the bundled
 * {@code cascades/haarcascade_frontalface_default.xml} from OpenCV 4.10) and
 * evaluates it against each candidate window of the input image across multiple
 * scales. The algorithm is the standard Viola-Jones pipeline:</p>
 *
 * <ol>
 *   <li>Convert the image to a grayscale byte buffer.</li>
 *   <li>Build an <em>integral image</em> (summed-area table) and a
 *       <em>squared integral image</em>, both in a single O(WH) pass, so that
 *       rectangle sums and window variance are later computable in O(1).</li>
 *   <li>Slide a detector window across the image at successively larger scales
 *       (start at the trained window size, grow by {@code scaleFactor} until
 *       larger than the image).</li>
 *   <li>At every window position, scale-normalise the cascade's Haar features
 *       to the current window size, compute window variance for feature
 *       normalisation, and evaluate the cascade stage by stage. A window is
 *       rejected as soon as any stage's summed weak-classifier response falls
 *       below that stage's threshold.</li>
 *   <li>Apply non-maximum suppression to merge overlapping detections into a
 *       final list of face regions, computing each merged region's
 *       confidence from the number of supporting detections.</li>
 * </ol>
 *
 * <p>The default cascade detects upright frontal faces down to 24×24 pixels at
 * roughly 90–95% true-positive rate on well-lit images. It is the same
 * detector used by the OpenCV reference implementation; we are only swapping
 * the evaluator from C++ to pure Java, not the trained weights.</p>
 *
 * <h3>Thread safety</h3>
 * <p>Instances are safe for concurrent use. The cascade data is immutable
 * after construction and detection state is allocated per call.</p>
 */
public final class HaarCascadeFaceDetector implements FaceDetector, Serializable {

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(HaarCascadeFaceDetector.class);

    /** Algorithm name reported to persistence and metrics. */
    public static final String ALGORITHM_NAME = "HaarCascade";

    /** Classpath location of the default frontal-face cascade bundled with the library. */
    public static final String DEFAULT_CASCADE_RESOURCE = "cascades/haarcascade_frontalface_default.xml";

    /** Default scale step between successive pyramid levels. */
    public static final double DEFAULT_SCALE_FACTOR = 1.1;

    /** Default minimum neighbour count required for a candidate to survive NMS. */
    public static final int DEFAULT_MIN_NEIGHBOURS = 3;

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
     * @param minNeighbours minimum number of overlapping raw detections required for a region to survive NMS
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
        log.debug("HaarCascadeFaceDetector initialized: {}, minFaceSize={}, scaleFactor={}, minNeighbours={}",
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
        if (width < cascade.getWindowWidth() || height < cascade.getWindowHeight()) {
            return Collections.emptyList();
        }

        // 1. Grayscale buffer + 2. integral + squared integral
        int[] grayscale = toGrayscale(src, width, height);
        long[] integral = new long[(width + 1) * (height + 1)];
        long[] squaredIntegral = new long[(width + 1) * (height + 1)];
        buildIntegralImages(grayscale, width, height, integral, squaredIntegral);

        // 3. Multi-scale sliding window
        List<RawDetection> raw = new ArrayList<>();
        double scale = (double) minFaceSize / cascade.getWindowWidth();
        double maxScaleX = (double) width / cascade.getWindowWidth();
        double maxScaleY = (double) height / cascade.getWindowHeight();
        double maxScale = Math.min(maxScaleX, maxScaleY);
        while (scale <= maxScale) {
            int windowW = (int) Math.round(cascade.getWindowWidth() * scale);
            int windowH = (int) Math.round(cascade.getWindowHeight() * scale);
            if (windowW < minFaceSize || windowH < minFaceSize) {
                scale *= scaleFactor;
                continue;
            }
            // Step by ~5% of the window size (OpenCV default), minimum 1 pixel.
            int step = Math.max(1, (int) Math.round(scale * 2.0));
            scanAtScale(integral, squaredIntegral, width, height, scale, windowW, windowH, step, raw);
            scale *= scaleFactor;
        }

        // 4. Non-maximum suppression
        List<FaceRegion> regions = mergeDetections(raw, minConfidence);
        log.debug("Haar detection: {} raw candidates -> {} regions", raw.size(), regions.size());
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

    /**
     * Sweeps the sliding window across the image at a fixed scale and records
     * every window that survives all cascade stages.
     */
    private void scanAtScale(long[] integral,
                             long[] squaredIntegral,
                             int imageWidth,
                             int imageHeight,
                             double scale,
                             int windowW,
                             int windowH,
                             int step,
                             List<RawDetection> out) {
        int stride = imageWidth + 1;
        int maxX = imageWidth - windowW;
        int maxY = imageHeight - windowH;
        int windowArea = windowW * windowH;
        // Inverse of the current window area — used to normalise every feature
        // value so comparisons against the (tiny, per-pixel) node thresholds in
        // the XML remain meaningful across scales. Computed once per scale
        // rather than once per window.
        double invWindowArea = 1.0 / windowArea;
        for (int y = 0; y <= maxY; y += step) {
            for (int x = 0; x <= maxX; x += step) {
                double variance = computeVariance(integral, squaredIntegral, stride, x, y, windowW, windowH, windowArea);
                if (variance < 1.0) {
                    // Uniform region — cannot contain a face.
                    continue;
                }
                double stddev = Math.sqrt(variance);
                if (evaluateCascade(integral, stride, x, y, scale, stddev, invWindowArea)) {
                    out.add(new RawDetection(x, y, windowW, windowH));
                }
            }
        }
    }

    /**
     * Runs the full cascade on the window at ({@code x}, {@code y}) at the
     * given {@code scale}. Returns {@code true} iff every stage's weighted
     * sum exceeds its rejection threshold.
     *
     * <p>The caller passes in the per-scale {@code invWindowArea} so we can
     * area-normalise each feature value into the same per-pixel units that the
     * node thresholds in the OpenCV XML are expressed in.</p>
     */
    private boolean evaluateCascade(long[] integral,
                                    int stride,
                                    int x,
                                    int y,
                                    double scale,
                                    double stddev,
                                    double invWindowArea) {
        HaarCascade.Stage[] stages = cascade.getStages();
        HaarCascade.Feature[] features = cascade.getFeatures();
        for (HaarCascade.Stage stage : stages) {
            double stageSum = 0.0;
            HaarCascade.WeakClassifier[] weaks = stage.getClassifiers();
            for (HaarCascade.WeakClassifier weak : weaks) {
                HaarCascade.Feature feature = features[weak.getFeatureIndex()];
                // Raw sum of weighted rectangle sums, divided by the current
                // window area so the result is in per-pixel intensity units.
                double featureValue = computeFeatureValue(integral, stride, x, y, feature, scale) * invWindowArea;
                // OpenCV stores node thresholds in per-pixel units and normalises
                // feature values by window stddev at runtime. Equivalent form
                // (one multiplication per weak classifier):
                //     featureValue  <  nodeThreshold * stddev
                double threshold = weak.getNodeThreshold() * stddev;
                stageSum += (featureValue < threshold) ? weak.getLeftLeaf() : weak.getRightLeaf();
            }
            if (stageSum < stage.getThreshold()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Computes the unnormalised Haar feature response at window origin
     * ({@code x}, {@code y}) for the given scale, using the integral image for
     * O(1) rectangle sums.
     *
     * <p>Rectangle coordinates inside each feature are in the cascade's trained
     * window coordinate system; we scale them up to the current window size
     * before reading the integral image. The caller is responsible for dividing
     * the returned value by the current window area.</p>
     */
    private double computeFeatureValue(long[] integral,
                                       int stride,
                                       int windowX,
                                       int windowY,
                                       HaarCascade.Feature feature,
                                       double scale) {
        double sum = 0.0;
        for (HaarCascade.Rect rect : feature.getRects()) {
            int rx = (int) Math.round(rect.getX() * scale);
            int ry = (int) Math.round(rect.getY() * scale);
            int rw = (int) Math.round(rect.getWidth() * scale);
            int rh = (int) Math.round(rect.getHeight() * scale);
            if (rw <= 0 || rh <= 0) {
                continue;
            }
            long rectSum = rectSum(integral, stride, windowX + rx, windowY + ry, rw, rh);
            sum += rectSum * rect.getWeight();
        }
        return sum;
    }

    /** Computes window variance using the integral and squared integral images. */
    private double computeVariance(long[] integral,
                                   long[] squared,
                                   int stride,
                                   int x,
                                   int y,
                                   int w,
                                   int h,
                                   int area) {
        long s = rectSum(integral, stride, x, y, w, h);
        long s2 = rectSum(squared, stride, x, y, w, h);
        double mean = s / (double) area;
        double variance = (s2 / (double) area) - mean * mean;
        return Math.max(0.0, variance);
    }

    /**
     * O(1) rectangle sum via summed-area table. The integral image is
     * {@code (width+1) * (height+1)} with a zero row + column at (0,0) so that
     * the four-corner formula handles windows touching the image edge without
     * special cases.
     */
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
     * Groups overlapping raw detections using OpenCV-style union-find by IoU,
     * then keeps only groups with at least {@link #minNeighbours} members.
     * The final region's coordinates are the average of the members'.
     */
    private List<FaceRegion> mergeDetections(List<RawDetection> raw, double minConfidence) {
        if (raw.isEmpty()) {
            return Collections.emptyList();
        }
        int n = raw.size();
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (iou(raw.get(i), raw.get(j)) >= 0.4) {
                    union(parent, i, j);
                }
            }
        }
        // Group by root
        java.util.Map<Integer, List<RawDetection>> groups = new java.util.HashMap<>();
        for (int i = 0; i < n; i++) {
            int root = find(parent, i);
            groups.computeIfAbsent(root, k -> new ArrayList<>()).add(raw.get(i));
        }
        List<FaceRegion> out = new ArrayList<>();
        int maxSupport = 1;
        for (List<RawDetection> group : groups.values()) {
            maxSupport = Math.max(maxSupport, group.size());
        }
        for (List<RawDetection> group : groups.values()) {
            if (group.size() < minNeighbours) {
                continue;
            }
            int sx = 0, sy = 0, sw = 0, sh = 0;
            for (RawDetection d : group) {
                sx += d.x;
                sy += d.y;
                sw += d.w;
                sh += d.h;
            }
            int size = group.size();
            int cx = sx / size;
            int cy = sy / size;
            int cw = sw / size;
            int ch = sh / size;
            // Confidence = fraction of the biggest group that supported this one.
            double confidence = Math.min(1.0, (double) size / (double) Math.max(1, maxSupport));
            if (confidence < minConfidence) {
                continue;
            }
            out.add(new FaceRegion(cx, cy, cw, ch, confidence));
        }
        return out;
    }

    private static int find(int[] parent, int i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }

    private static void union(int[] parent, int a, int b) {
        int ra = find(parent, a);
        int rb = find(parent, b);
        if (ra != rb) {
            parent[ra] = rb;
        }
    }

    private static double iou(RawDetection a, RawDetection b) {
        int x1 = Math.max(a.x, b.x);
        int y1 = Math.max(a.y, b.y);
        int x2 = Math.min(a.x + a.w, b.x + b.w);
        int y2 = Math.min(a.y + a.h, b.y + b.h);
        if (x1 >= x2 || y1 >= y2) {
            return 0.0;
        }
        int inter = (x2 - x1) * (y2 - y1);
        int union = a.w * a.h + b.w * b.h - inter;
        return (double) inter / union;
    }

    /** Lightweight value object for raw sliding-window detections prior to NMS. */
    private static final class RawDetection {
        final int x;
        final int y;
        final int w;
        final int h;

        RawDetection(int x, int y, int w, int h) {
            this.x = x;
            this.y = y;
            this.w = w;
            this.h = h;
        }
    }
}
