package com.facerecognition.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

import com.facerecognition.domain.service.FaceClassifier.DistanceMetric;

/**
 * Central, type-safe configuration for the face-recognition pipeline.
 *
 * <p>Every runtime knob lives here and is bound from {@code application.yml}
 * under the {@code facerecognition.*} prefix. The {@link FaceRecognitionAutoConfiguration}
 * reads these properties and wires the detector, extractor, classifier, and
 * service beans accordingly, so there is a single source of truth for
 * everything from the default algorithm to rate-limit thresholds.</p>
 */
@ConfigurationProperties(prefix = "facerecognition")
public class FaceRecognitionProperties {

    private final Detection detection = new Detection();
    private final Extraction extraction = new Extraction();
    private final Classification classification = new Classification();
    private final Recognition recognition = new Recognition();
    private final Quality quality = new Quality();
    private final Image image = new Image();
    private final Model model = new Model();
    private final RateLimit ratelimit = new RateLimit();
    private final Cors cors = new Cors();
    private final Security security = new Security();

    public Detection getDetection() { return detection; }
    public Extraction getExtraction() { return extraction; }
    public Classification getClassification() { return classification; }
    public Recognition getRecognition() { return recognition; }
    public Quality getQuality() { return quality; }
    public Image getImage() { return image; }
    public Model getModel() { return model; }
    public RateLimit getRatelimit() { return ratelimit; }
    public Cors getCors() { return cors; }
    public Security getSecurity() { return security; }

    /** Face-detection tuning. */
    public static class Detection {
        /** Which detector implementation to use. */
        private DetectorType type = DetectorType.HAAR_CASCADE;
        /** Minimum face side in pixels. */
        private int minFaceSize = 30;
        /** Minimum detector confidence in [0, 1]. */
        private double minConfidence = 0.5;
        /** Maximum faces returned per image. */
        private int maxFaces = 10;
        /** Scale step between successive pyramid levels (Haar only). */
        private double scaleFactor = 1.1;
        /** Minimum number of overlapping raw detections required after NMS (Haar only). */
        private int minNeighbours = 3;

        public DetectorType getType() { return type; }
        public void setType(DetectorType type) { this.type = type; }
        public int getMinFaceSize() { return minFaceSize; }
        public void setMinFaceSize(int minFaceSize) { this.minFaceSize = minFaceSize; }
        public double getMinConfidence() { return minConfidence; }
        public void setMinConfidence(double minConfidence) { this.minConfidence = minConfidence; }
        public int getMaxFaces() { return maxFaces; }
        public void setMaxFaces(int maxFaces) { this.maxFaces = maxFaces; }
        public double getScaleFactor() { return scaleFactor; }
        public void setScaleFactor(double scaleFactor) { this.scaleFactor = scaleFactor; }
        public int getMinNeighbours() { return minNeighbours; }
        public void setMinNeighbours(int minNeighbours) { this.minNeighbours = minNeighbours; }
    }

    /** Built-in detector implementations. */
    public enum DetectorType {
        /** Real Viola-Jones detector using an OpenCV Haar cascade (default). */
        HAAR_CASCADE,
        /** Skin-colour heuristic — fast fallback, not production-grade. */
        SKIN_COLOR
    }

    /** Feature-extraction tuning. */
    public static class Extraction {
        /** Which extractor to use. */
        private ExtractorType algorithm = ExtractorType.EIGENFACES;
        /** Number of components / dimensions. */
        private int numComponents = 10;
        /** LBPH-specific configuration (ignored for other extractors). */
        private final Lbph lbph = new Lbph();
        /** ONNX-specific configuration (ignored unless algorithm=onnx). */
        private final Onnx onnx = new Onnx();

        public ExtractorType getAlgorithm() { return algorithm; }
        public void setAlgorithm(ExtractorType algorithm) { this.algorithm = algorithm; }
        public int getNumComponents() { return numComponents; }
        public void setNumComponents(int numComponents) { this.numComponents = numComponents; }
        public Lbph getLbph() { return lbph; }
        public Onnx getOnnx() { return onnx; }

        public static class Lbph {
            private int gridX = 8;
            private int gridY = 8;
            private int radius = 1;
            private int neighbors = 8;

            public int getGridX() { return gridX; }
            public void setGridX(int gridX) { this.gridX = gridX; }
            public int getGridY() { return gridY; }
            public void setGridY(int gridY) { this.gridY = gridY; }
            public int getRadius() { return radius; }
            public void setRadius(int radius) { this.radius = radius; }
            public int getNeighbors() { return neighbors; }
            public void setNeighbors(int neighbors) { this.neighbors = neighbors; }
        }

        public static class Onnx {
            /** Filesystem path or classpath URL to a .onnx model. Empty disables the backend. */
            private String modelPath = "";
            /** Execution provider, e.g. {@code cpu}, {@code cuda}, {@code coreml}. */
            private String provider = "cpu";
            /** Expected embedding dimension produced by the model. */
            private int embeddingDimension = 128;
            /** Input image side expected by the model. */
            private int inputSize = 160;

            public String getModelPath() { return modelPath; }
            public void setModelPath(String modelPath) { this.modelPath = modelPath; }
            public String getProvider() { return provider; }
            public void setProvider(String provider) { this.provider = provider; }
            public int getEmbeddingDimension() { return embeddingDimension; }
            public void setEmbeddingDimension(int embeddingDimension) { this.embeddingDimension = embeddingDimension; }
            public int getInputSize() { return inputSize; }
            public void setInputSize(int inputSize) { this.inputSize = inputSize; }
        }
    }

    /** Built-in extractor implementations. */
    public enum ExtractorType {
        /** Turk & Pentland, Eigenfaces (PCA). */
        EIGENFACES,
        /** Belhumeur et al., Fisherfaces (LDA). */
        FISHERFACES,
        /** Ahonen et al., LBP Histograms. */
        LBPH,
        /** ONNX-based deep embedding (bring your own model). */
        ONNX
    }

    /** Classifier tuning. */
    public static class Classification {
        private ClassifierType algorithm = ClassifierType.KNN;
        private int kNeighbors = 3;
        private DistanceMetric distanceMetric = DistanceMetric.EUCLIDEAN;
        private boolean useAverageFeatures = false;

        public ClassifierType getAlgorithm() { return algorithm; }
        public void setAlgorithm(ClassifierType algorithm) { this.algorithm = algorithm; }
        public int getKNeighbors() { return kNeighbors; }
        public void setKNeighbors(int kNeighbors) { this.kNeighbors = kNeighbors; }
        public DistanceMetric getDistanceMetric() { return distanceMetric; }
        public void setDistanceMetric(DistanceMetric distanceMetric) { this.distanceMetric = distanceMetric; }
        public boolean isUseAverageFeatures() { return useAverageFeatures; }
        public void setUseAverageFeatures(boolean useAverageFeatures) { this.useAverageFeatures = useAverageFeatures; }
    }

    /** Built-in classifier implementations. */
    public enum ClassifierType {
        /** k-Nearest Neighbours. */
        KNN
    }

    /** Recognition-pipeline knobs. */
    public static class Recognition {
        /** Minimum classifier confidence in [0,1] for a recognised match. */
        private double threshold = 0.6;
        /** Maximum alternatives returned alongside the best match. */
        private int maxAlternatives = 5;

        public double getThreshold() { return threshold; }
        public void setThreshold(double threshold) { this.threshold = threshold; }
        public int getMaxAlternatives() { return maxAlternatives; }
        public void setMaxAlternatives(int maxAlternatives) { this.maxAlternatives = maxAlternatives; }
    }

    /** Image-quality gate. */
    public static class Quality {
        /** Minimum quality score [0,1] required for enrolment. */
        private double minScore = 0.3;
        /** Whether to apply the gate at enrolment time. */
        private boolean validateOnEnroll = true;

        public double getMinScore() { return minScore; }
        public void setMinScore(double minScore) { this.minScore = minScore; }
        public boolean isValidateOnEnroll() { return validateOnEnroll; }
        public void setValidateOnEnroll(boolean validateOnEnroll) { this.validateOnEnroll = validateOnEnroll; }
    }

    /** Image-preprocessing knobs. */
    public static class Image {
        private int targetWidth = 100;
        private int targetHeight = 100;
        private boolean histogramEqualization = true;
        private boolean faceAlignment = true;

        public int getTargetWidth() { return targetWidth; }
        public void setTargetWidth(int targetWidth) { this.targetWidth = targetWidth; }
        public int getTargetHeight() { return targetHeight; }
        public void setTargetHeight(int targetHeight) { this.targetHeight = targetHeight; }
        public boolean isHistogramEqualization() { return histogramEqualization; }
        public void setHistogramEqualization(boolean histogramEqualization) { this.histogramEqualization = histogramEqualization; }
        public boolean isFaceAlignment() { return faceAlignment; }
        public void setFaceAlignment(boolean faceAlignment) { this.faceAlignment = faceAlignment; }
    }

    /** Model persistence. */
    public static class Model {
        /** Enable auto-save after a successful training run. */
        private boolean autoSave = true;
        /** Enable auto-load on application startup. */
        private boolean autoLoad = true;
        /** Target file path for saved models. */
        private String savePath = "data/models/default.frm";

        public boolean isAutoSave() { return autoSave; }
        public void setAutoSave(boolean autoSave) { this.autoSave = autoSave; }
        public boolean isAutoLoad() { return autoLoad; }
        public void setAutoLoad(boolean autoLoad) { this.autoLoad = autoLoad; }
        public String getSavePath() { return savePath; }
        public void setSavePath(String savePath) { this.savePath = savePath; }
    }

    /** Per-IP token-bucket rate limiter. */
    public static class RateLimit {
        private boolean enabled = true;
        private int requestsPerMinute = 60;

        public boolean isEnabled() { return enabled; }
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        public int getRequestsPerMinute() { return requestsPerMinute; }
        public void setRequestsPerMinute(int requestsPerMinute) { this.requestsPerMinute = requestsPerMinute; }
    }

    /** Cross-Origin Resource Sharing configuration. */
    public static class Cors {
        private boolean enabled = false;
        private String allowedOrigins = "*";
        private String allowedMethods = "GET,POST,PATCH,DELETE,OPTIONS";
        private String allowedHeaders = "*";
        private long maxAgeSeconds = 3600;

        public boolean isEnabled() { return enabled; }
        public void setEnabled(boolean enabled) { this.enabled = enabled; }
        public String getAllowedOrigins() { return allowedOrigins; }
        public void setAllowedOrigins(String allowedOrigins) { this.allowedOrigins = allowedOrigins; }
        public String getAllowedMethods() { return allowedMethods; }
        public void setAllowedMethods(String allowedMethods) { this.allowedMethods = allowedMethods; }
        public String getAllowedHeaders() { return allowedHeaders; }
        public void setAllowedHeaders(String allowedHeaders) { this.allowedHeaders = allowedHeaders; }
        public long getMaxAgeSeconds() { return maxAgeSeconds; }
        public void setMaxAgeSeconds(long maxAgeSeconds) { this.maxAgeSeconds = maxAgeSeconds; }
    }

    /** Minimal API-key auth (opt-in). */
    public static class Security {
        /** If non-blank, requests must carry {@code X-API-Key} with this value. */
        private String apiKey = "";
        /** Header name to read the API key from. */
        private String apiKeyHeader = "X-API-Key";

        public String getApiKey() { return apiKey; }
        public void setApiKey(String apiKey) { this.apiKey = apiKey; }
        public String getApiKeyHeader() { return apiKeyHeader; }
        public void setApiKeyHeader(String apiKeyHeader) { this.apiKeyHeader = apiKeyHeader; }
    }
}
