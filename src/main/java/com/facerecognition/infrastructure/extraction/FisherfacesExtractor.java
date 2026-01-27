package com.facerecognition.infrastructure.extraction;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.domain.service.FeatureExtractor.ExtractorConfig;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

import java.io.Serializable;
import java.util.*;

/**
 * Fisherfaces (LDA) feature extractor implementation.
 *
 * <p>Fisherfaces uses Linear Discriminant Analysis to find a projection
 * that maximizes the ratio of between-class scatter to within-class scatter.
 * This makes it more robust to lighting variations than Eigenfaces.</p>
 *
 * <h3>Algorithm Overview:</h3>
 * <ol>
 *   <li>Apply PCA to reduce dimensionality (to N-c dimensions)</li>
 *   <li>Compute within-class scatter matrix Sw</li>
 *   <li>Compute between-class scatter matrix Sb</li>
 *   <li>Find eigenvectors of Sw⁻¹Sb</li>
 *   <li>Select top K eigenvectors (fisherfaces)</li>
 * </ol>
 *
 * <h3>Mathematical Formulation:</h3>
 * <pre>
 * Within-class scatter: Sw = Σc Σᵢ (xᵢ - μc)(xᵢ - μc)ᵀ
 * Between-class scatter: Sb = Σc nc(μc - μ)(μc - μ)ᵀ
 * Optimization: max |WᵀSbW| / |WᵀSwW|
 * Solution: eigenvectors of Sw⁻¹Sb
 * </pre>
 *
 * <h3>Requirements:</h3>
 * <ul>
 *   <li>Multiple samples per class (identity)</li>
 *   <li>At least 2 classes for training</li>
 *   <li>Number of features limited to c-1 (number of classes - 1)</li>
 * </ul>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see FeatureExtractor
 * @see EigenfacesExtractor
 */
public class FisherfacesExtractor implements FeatureExtractor, Serializable {

    private static final long serialVersionUID = 2L;

    public static final String ALGORITHM_NAME = "Fisherfaces";
    public static final int VERSION = 2;
    private static final double REGULARIZATION = 1e-6;

    private final ExtractorConfig config;
    private boolean trained;

    // Model components
    private Matrix meanFace;
    private Matrix projectionMatrix; // Combined PCA + LDA projection
    private int numComponents;
    private Map<String, Integer> labelMap;
    private List<String> labels;

    /**
     * Creates a Fisherfaces extractor with default settings.
     */
    public FisherfacesExtractor() {
        this(new ExtractorConfig().setNumComponents(10));
    }

    /**
     * Creates a Fisherfaces extractor with specified number of components.
     *
     * @param numComponents the number of fisherface components to use
     */
    public FisherfacesExtractor(int numComponents) {
        this(new ExtractorConfig().setNumComponents(numComponents));
    }

    /**
     * Creates a Fisherfaces extractor with custom configuration.
     *
     * @param config the extractor configuration
     */
    public FisherfacesExtractor(ExtractorConfig config) {
        this.config = Objects.requireNonNull(config);
        this.trained = false;
        this.labelMap = new HashMap<>();
        this.labels = new ArrayList<>();
    }

    @Override
    public void train(List<FaceImage> faces, List<String> labels) {
        if (trained) {
            throw new IllegalStateException("Extractor already trained. Call reset() first.");
        }
        if (faces == null || faces.isEmpty()) {
            throw new IllegalArgumentException("Training set cannot be empty");
        }
        if (labels == null || labels.size() != faces.size()) {
            throw new IllegalArgumentException("Labels must be provided for Fisherfaces");
        }

        // Build label mapping
        buildLabelMap(labels);
        int numClasses = labelMap.size();

        if (numClasses < 2) {
            throw new IllegalArgumentException("At least 2 classes required for Fisherfaces");
        }

        // Convert faces to pixel matrix
        double[][] pixelMatrix = constructImageMatrix(faces);
        int numSamples = pixelMatrix.length;
        int numPixels = pixelMatrix[0].length;

        // Step 1: Compute mean face
        computeMeanFace(pixelMatrix);

        // Step 2: Center the data
        Matrix centered = centerData(pixelMatrix);

        // Step 3: PCA to reduce to N-c dimensions
        int pcaComponents = numSamples - numClasses;
        Matrix pcaProjection = computePCAProjection(centered, pcaComponents);

        // Step 4: Project data to PCA space
        Matrix pcaData = pcaProjection.transpose().times(centered);

        // Step 5: Compute scatter matrices in PCA space
        int[] labelIndices = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            labelIndices[i] = labelMap.get(labels.get(i));
        }

        Matrix[] scatters = computeScatterMatrices(pcaData, labelIndices, numClasses);
        Matrix Sw = scatters[0];
        Matrix Sb = scatters[1];

        // Step 6: Regularize Sw and compute LDA
        Matrix SwReg = regularize(Sw);
        Matrix SwInv;
        try {
            SwInv = SwReg.inverse();
        } catch (RuntimeException e) {
            // Matrix may still be singular despite regularization (degenerate data)
            // Apply stronger regularization as fallback
            double strongerReg = REGULARIZATION * 1000;
            Matrix identity = Matrix.identity(SwReg.getRowDimension(), SwReg.getColumnDimension());
            SwReg = Sw.plus(identity.times(strongerReg));
            try {
                SwInv = SwReg.inverse();
            } catch (RuntimeException e2) {
                throw new IllegalStateException(
                    "Cannot compute Fisherfaces: within-class scatter matrix is singular. " +
                    "This may occur with degenerate training data (identical faces or too few samples). " +
                    "Try adding more diverse training samples or use Eigenfaces instead.", e2);
            }
        }
        Matrix ldaMatrix = SwInv.times(Sb);

        // Step 7: Compute eigenvectors
        EigenvalueDecomposition eigen = ldaMatrix.eig();
        Matrix ldaEigenvectors = extractRealEigenvectors(eigen);

        // Step 8: Select top components (limited to numClasses - 1)
        numComponents = Math.min(config.getNumComponents(), numClasses - 1);
        Matrix ldaProjection = ldaEigenvectors.getMatrix(0, ldaEigenvectors.getRowDimension() - 1,
                                                          0, numComponents - 1);

        // Step 9: Combine PCA and LDA projections
        projectionMatrix = pcaProjection.times(ldaProjection);

        trained = true;
        this.labels = new ArrayList<>(labels);
    }

    private void buildLabelMap(List<String> labels) {
        labelMap.clear();
        int index = 0;
        for (String label : labels) {
            if (!labelMap.containsKey(label)) {
                labelMap.put(label, index++);
            }
        }
    }

    private double[][] constructImageMatrix(List<FaceImage> faces) {
        int width = config.getImageWidth();
        int height = config.getImageHeight();

        double[][] matrix = new double[faces.size()][width * height];
        for (int i = 0; i < faces.size(); i++) {
            FaceImage face = faces.get(i);
            FaceImage resized = face.getWidth() != width || face.getHeight() != height
                ? face.resize(width, height)
                : face;
            matrix[i] = resized.toGrayscaleArray();
        }
        return matrix;
    }

    private void computeMeanFace(double[][] pixelMatrix) {
        int numSamples = pixelMatrix.length;
        int numPixels = pixelMatrix[0].length;

        double[] mean = new double[numPixels];
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numPixels; j++) {
                mean[j] += pixelMatrix[i][j];
            }
        }
        for (int j = 0; j < numPixels; j++) {
            mean[j] /= numSamples;
        }

        meanFace = new Matrix(mean, 1).transpose();
    }

    private Matrix centerData(double[][] pixelMatrix) {
        int numSamples = pixelMatrix.length;
        int numPixels = pixelMatrix[0].length;

        Matrix matrix = new Matrix(pixelMatrix).transpose(); // numPixels x numSamples

        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numPixels; j++) {
                matrix.set(j, i, matrix.get(j, i) - meanFace.get(j, 0));
            }
        }

        return matrix;
    }

    private Matrix computePCAProjection(Matrix centered, int numComponents) {
        int numPixels = centered.getRowDimension();
        int numSamples = centered.getColumnDimension();

        // Compute covariance using efficient trick
        Matrix At = centered.transpose();
        Matrix L = At.times(centered);

        EigenvalueDecomposition eigen = L.eig();
        Matrix eigenVectors = eigen.getV();
        Matrix eigenValues = eigen.getD();

        // Sort by eigenvalue
        double[] values = new double[numSamples];
        int[] indices = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            values[i] = eigenValues.get(i, i);
            indices[i] = i;
        }

        for (int i = 0; i < numSamples - 1; i++) {
            for (int j = i + 1; j < numSamples; j++) {
                if (values[j] > values[i]) {
                    double tv = values[i]; values[i] = values[j]; values[j] = tv;
                    int ti = indices[i]; indices[i] = indices[j]; indices[j] = ti;
                }
            }
        }

        // Select top components
        int selectedComponents = Math.min(numComponents, numSamples);
        Matrix sortedVectors = new Matrix(numSamples, selectedComponents);
        for (int i = 0; i < selectedComponents; i++) {
            for (int j = 0; j < numSamples; j++) {
                sortedVectors.set(j, i, eigenVectors.get(j, indices[i]));
            }
        }

        // Project to pixel space
        Matrix pcaVectors = centered.times(sortedVectors);

        // Normalize
        for (int i = 0; i < selectedComponents; i++) {
            double norm = 0;
            for (int j = 0; j < numPixels; j++) {
                norm += pcaVectors.get(j, i) * pcaVectors.get(j, i);
            }
            norm = Math.sqrt(norm);
            if (norm > 0) {
                for (int j = 0; j < numPixels; j++) {
                    pcaVectors.set(j, i, pcaVectors.get(j, i) / norm);
                }
            }
        }

        return pcaVectors;
    }

    private Matrix[] computeScatterMatrices(Matrix data, int[] labelIndices, int numClasses) {
        int dim = data.getRowDimension();
        int numSamples = data.getColumnDimension();

        // Compute class means
        Matrix[] classMeans = new Matrix[numClasses];
        int[] classCounts = new int[numClasses];

        for (int c = 0; c < numClasses; c++) {
            classMeans[c] = new Matrix(dim, 1);
        }

        for (int i = 0; i < numSamples; i++) {
            int c = labelIndices[i];
            for (int j = 0; j < dim; j++) {
                classMeans[c].set(j, 0, classMeans[c].get(j, 0) + data.get(j, i));
            }
            classCounts[c]++;
        }

        for (int c = 0; c < numClasses; c++) {
            if (classCounts[c] > 0) {
                classMeans[c] = classMeans[c].times(1.0 / classCounts[c]);
            }
        }

        // Compute overall mean
        Matrix overallMean = new Matrix(dim, 1);
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < dim; j++) {
                overallMean.set(j, 0, overallMean.get(j, 0) + data.get(j, i));
            }
        }
        overallMean = overallMean.times(1.0 / numSamples);

        // Within-class scatter
        Matrix Sw = new Matrix(dim, dim);
        for (int i = 0; i < numSamples; i++) {
            int c = labelIndices[i];
            Matrix diff = data.getMatrix(0, dim - 1, i, i).minus(classMeans[c]);
            Sw = Sw.plus(diff.times(diff.transpose()));
        }

        // Between-class scatter
        Matrix Sb = new Matrix(dim, dim);
        for (int c = 0; c < numClasses; c++) {
            Matrix diff = classMeans[c].minus(overallMean);
            Sb = Sb.plus(diff.times(diff.transpose()).times(classCounts[c]));
        }

        return new Matrix[]{Sw, Sb};
    }

    private Matrix regularize(Matrix Sw) {
        int dim = Sw.getRowDimension();
        Matrix identity = Matrix.identity(dim, dim);
        return Sw.plus(identity.times(REGULARIZATION));
    }

    private Matrix extractRealEigenvectors(EigenvalueDecomposition eigen) {
        Matrix V = eigen.getV();
        Matrix D = eigen.getD();
        int n = V.getColumnDimension();

        // Get eigenvalues and sort by real part (descending)
        double[] realParts = new double[n];
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            realParts[i] = D.get(i, i);
            indices[i] = i;
        }

        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (realParts[j] > realParts[i]) {
                    double tv = realParts[i]; realParts[i] = realParts[j]; realParts[j] = tv;
                    int ti = indices[i]; indices[i] = indices[j]; indices[j] = ti;
                }
            }
        }

        Matrix sorted = new Matrix(V.getRowDimension(), n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < V.getRowDimension(); j++) {
                sorted.set(j, i, V.get(j, indices[i]));
            }
        }

        return sorted;
    }

    @Override
    public FeatureVector extract(FaceImage face) {
        if (!trained) {
            throw new IllegalStateException("Extractor not trained");
        }

        FaceImage resized = face.getWidth() != config.getImageWidth() ||
                           face.getHeight() != config.getImageHeight()
            ? face.resize(config.getImageWidth(), config.getImageHeight())
            : face;

        double[] pixels = resized.toGrayscaleArray();
        Matrix faceVector = new Matrix(pixels, pixels.length);
        Matrix centered = faceVector.minus(meanFace);

        Matrix projection = projectionMatrix.transpose().times(centered);

        double[] coefficients = new double[numComponents];
        for (int i = 0; i < numComponents; i++) {
            coefficients[i] = projection.get(i, 0);
        }

        if (config.isNormalize()) {
            double norm = 0;
            for (double c : coefficients) norm += c * c;
            norm = Math.sqrt(norm);
            if (norm > 0) {
                for (int i = 0; i < coefficients.length; i++) {
                    coefficients[i] /= norm;
                }
            }
        }

        return new FeatureVector(coefficients, ALGORITHM_NAME, VERSION);
    }

    @Override
    public boolean isTrained() {
        return trained;
    }

    @Override
    public int getFeatureDimension() {
        return numComponents;
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
        return new int[]{config.getImageWidth(), config.getImageHeight()};
    }

    @Override
    public void reset() {
        trained = false;
        meanFace = null;
        projectionMatrix = null;
        numComponents = 0;
        labelMap.clear();
        labels.clear();
    }

    @Override
    public ExtractorConfig getConfig() {
        return config;
    }

    /**
     * Gets the number of classes used in training.
     *
     * @return the number of classes
     */
    public int getNumClasses() {
        return labelMap.size();
    }

    @Override
    public String toString() {
        return String.format("FisherfacesExtractor{components=%d, classes=%d, trained=%s}",
            numComponents, labelMap.size(), trained);
    }
}
