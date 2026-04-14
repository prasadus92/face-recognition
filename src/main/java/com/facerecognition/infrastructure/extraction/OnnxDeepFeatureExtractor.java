package com.facerecognition.infrastructure.extraction;

import java.io.Serializable;
import java.util.List;
import java.util.Objects;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.service.FeatureExtractor;

/**
 * Experimental deep-learning feature extractor scaffold.
 *
 * <p>This class defines the contract and wiring for a modern embedding backend
 * (FaceNet, ArcFace, InsightFace's {@code buffalo_s}, …) running via ONNX Runtime,
 * but <b>does not itself ship a model</b>. Activating it requires:</p>
 *
 * <ol>
 *   <li>Adding a runtime-scope dependency on {@code com.microsoft.onnxruntime:onnxruntime}.</li>
 *   <li>Setting {@code facerecognition.extraction.algorithm=onnx} in {@code application.yml}.</li>
 *   <li>Pointing {@code facerecognition.extraction.onnx.model-path} at a
 *       {@code .onnx} file whose input is a normalised RGB tensor of size
 *       {@code (1, 3, inputSize, inputSize)} and whose output is a 1×D float embedding.</li>
 * </ol>
 *
 * <p>Until those steps are taken the extractor refuses to start, so the rest of the
 * service continues to work with the classical extractors. This is a deliberate
 * choice — bundling model weights changes the project's licensing story and download
 * footprint, and I'd rather keep the scaffold honest.</p>
 *
 * <p>See {@code docs/onnx.md} for the full integration notes.</p>
 */
public class OnnxDeepFeatureExtractor implements FeatureExtractor, Serializable {

    private static final long serialVersionUID = 1L;
    private static final Logger log = LoggerFactory.getLogger(OnnxDeepFeatureExtractor.class);

    /** Algorithm name reported to persistence and metrics. */
    public static final String ALGORITHM_NAME = "ONNX-Deep";

    /** Current algorithm version. Bump when the embedding space changes. */
    public static final int VERSION = 1;

    private final String modelPath;
    private final int embeddingDimension;
    private final int inputSize;
    private final String provider;
    private final ExtractorConfig config;

    private transient boolean modelLoaded;

    public OnnxDeepFeatureExtractor(String modelPath,
                                    int embeddingDimension,
                                    int inputSize,
                                    String provider) {
        this.modelPath = modelPath == null ? "" : modelPath;
        if (embeddingDimension <= 0) {
            throw new IllegalArgumentException("embeddingDimension must be > 0");
        }
        if (inputSize <= 0) {
            throw new IllegalArgumentException("inputSize must be > 0");
        }
        this.embeddingDimension = embeddingDimension;
        this.inputSize = inputSize;
        this.provider = Objects.requireNonNullElse(provider, "cpu");
        this.config = new ExtractorConfig()
                .setNumComponents(embeddingDimension)
                .setImageWidth(inputSize)
                .setImageHeight(inputSize)
                .setNormalize(true);
        this.modelLoaded = false;
        logConfiguration();
    }

    private void logConfiguration() {
        if (modelPath.isEmpty()) {
            log.warn("OnnxDeepFeatureExtractor configured without a model path. "
                    + "Set facerecognition.extraction.onnx.model-path and add an ONNX Runtime "
                    + "dependency to enable deep embeddings.");
        } else {
            log.info("OnnxDeepFeatureExtractor: modelPath={}, embeddingDim={}, inputSize={}, provider={}",
                    modelPath, embeddingDimension, inputSize, provider);
        }
    }

    @Override
    public void train(List<FaceImage> faces, List<String> labels) {
        // Deep embeddings don't require training over the user dataset; the
        // weights are fixed. We still need the model to be loadable, though.
        ensureModelLoaded();
    }

    @Override
    public boolean isTrained() {
        return modelLoaded;
    }

    @Override
    public FeatureVector extract(FaceImage face) {
        ensureModelLoaded();
        throw new UnsupportedOperationException(
                "OnnxDeepFeatureExtractor scaffold: add com.microsoft.onnxruntime:onnxruntime "
                        + "and implement ONNX inference in this class before using the ONNX backend.");
    }

    @Override
    public int getFeatureDimension() {
        return embeddingDimension;
    }

    @Override
    public String getAlgorithmName() {
        return ALGORITHM_NAME;
    }

    @Override
    public int getVersion() {
        return VERSION;
    }

    @Override
    public int[] getExpectedImageSize() {
        return new int[]{inputSize, inputSize};
    }

    @Override
    public void reset() {
        this.modelLoaded = false;
    }

    @Override
    public ExtractorConfig getConfig() {
        return config;
    }

    public String getModelPath() {
        return modelPath;
    }

    public String getProvider() {
        return provider;
    }

    private void ensureModelLoaded() {
        if (modelLoaded) {
            return;
        }
        if (modelPath.isEmpty()) {
            throw new IllegalStateException(
                    "ONNX backend selected but facerecognition.extraction.onnx.model-path is blank. "
                            + "Point it at a .onnx file and restart.");
        }
        // The scaffold does not actually attempt to load the model; a future
        // PR will add ONNX Runtime and implement real inference here.
        modelLoaded = true;
    }
}
