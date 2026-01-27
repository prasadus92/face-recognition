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
 * Eigenfaces (PCA) feature extractor implementation.
 *
 * <p>Eigenfaces uses Principal Component Analysis to find the
 * directions of maximum variance in a training set of faces.
 * These directions (eigenvectors) form a basis for representing
 * faces as linear combinations.</p>
 *
 * <h3>Algorithm Overview:</h3>
 * <ol>
 *   <li>Compute mean face from training set</li>
 *   <li>Subtract mean from each face</li>
 *   <li>Compute covariance matrix</li>
 *   <li>Find eigenvectors of covariance matrix</li>
 *   <li>Select top K eigenvectors (eigenfaces)</li>
 *   <li>Project faces onto eigenface space</li>
 * </ol>
 *
 * <h3>Mathematical Formulation:</h3>
 * <pre>
 * Given training faces X = [x₁, x₂, ..., xₙ]
 * Mean face: μ = (1/n) Σ xᵢ
 * Centered faces: Φᵢ = xᵢ - μ
 * Covariance: C = (1/n) Σ ΦᵢΦᵢᵀ
 * Eigenfaces: V = eigenvectors(C)
 * Projection: w = Vᵀ(x - μ)
 * </pre>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * EigenfacesExtractor extractor = new EigenfacesExtractor(10);
 * extractor.train(trainingFaces, null);
 * FeatureVector features = extractor.extract(probeFace);
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 1.0
 * @see FeatureExtractor
 */
public class EigenfacesExtractor implements FeatureExtractor, Serializable {

    private static final long serialVersionUID = 2L;

    /** Algorithm name for identification. */
    public static final String ALGORITHM_NAME = "Eigenfaces";

    /** Current algorithm version. */
    public static final int VERSION = 2;

    /** Default number of eigenfaces to use. */
    public static final int DEFAULT_NUM_COMPONENTS = 10;

    /** Minimum eigenvalue threshold for component selection. */
    private static final double EIGENVALUE_THRESHOLD = 0.0001;

    private final ExtractorConfig config;
    private boolean trained;

    // Trained model components
    private Matrix meanFace;
    private Matrix eigenVectors;
    private Matrix eigenValues;
    private int numEigenVectors;
    private double[] explainedVarianceRatio;

    /**
     * Creates an Eigenfaces extractor with default settings.
     */
    public EigenfacesExtractor() {
        this(new ExtractorConfig().setNumComponents(DEFAULT_NUM_COMPONENTS));
    }

    /**
     * Creates an Eigenfaces extractor with specified number of components.
     *
     * @param numComponents the number of eigenfaces to compute
     */
    public EigenfacesExtractor(int numComponents) {
        this(new ExtractorConfig().setNumComponents(numComponents));
    }

    /**
     * Creates an Eigenfaces extractor with custom configuration.
     *
     * @param config the extractor configuration
     */
    public EigenfacesExtractor(ExtractorConfig config) {
        this.config = Objects.requireNonNull(config, "Config cannot be null");
        this.trained = false;
    }

    @Override
    public void train(List<FaceImage> faces, List<String> labels) {
        if (trained) {
            throw new IllegalStateException("Extractor already trained. Call reset() first.");
        }
        if (faces == null || faces.isEmpty()) {
            throw new IllegalArgumentException("Training set cannot be empty");
        }

        // Convert faces to pixel matrix
        double[][] pixelMatrix = constructImageMatrix(faces);
        int numSamples = pixelMatrix.length;
        int numPixels = pixelMatrix[0].length;

        // Step 1: Compute mean face
        computeMeanFace(pixelMatrix);

        // Step 2: Center the data (subtract mean)
        Matrix centeredData = centerData(pixelMatrix);

        // Step 3: Compute covariance matrix (using efficient trick for high-dimensional data)
        Matrix covarianceMatrix = computeCovarianceMatrix(centeredData);

        // Step 4: Compute eigenvalues and eigenvectors
        EigenvalueDecomposition eigen = covarianceMatrix.eig();
        Matrix tempEigenValues = eigen.getD();
        Matrix tempEigenVectors = eigen.getV();

        // Step 5: Sort eigenvectors by eigenvalue (descending)
        sortEigenvectors(tempEigenValues, tempEigenVectors);

        // Step 6: Project eigenvectors to original space (if using efficient computation)
        if (numSamples < numPixels) {
            projectEigenvectorsToOriginalSpace(centeredData);
        }

        // Step 7: Normalize eigenvectors and compute variance ratios
        normalizeAndSelectComponents();

        trained = true;
    }

    private double[][] constructImageMatrix(List<FaceImage> faces) {
        int width = config.getImageWidth();
        int height = config.getImageHeight();
        int numPixels = width * height;

        double[][] matrix = new double[faces.size()][numPixels];
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

        Matrix matrix = new Matrix(pixelMatrix);
        Matrix meanRow = meanFace.transpose();

        // Subtract mean from each row
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numPixels; j++) {
                matrix.set(i, j, matrix.get(i, j) - meanRow.get(0, j));
            }
        }

        return matrix.transpose(); // Return as numPixels x numSamples
    }

    private Matrix computeCovarianceMatrix(Matrix centeredData) {
        int numPixels = centeredData.getRowDimension();
        int numSamples = centeredData.getColumnDimension();

        // Use the efficient trick: if numSamples < numPixels
        // compute L = A'A instead of C = AA' and then project
        if (numSamples < numPixels) {
            // L = A'A (smaller matrix: numSamples x numSamples)
            Matrix At = centeredData.transpose();
            return At.times(centeredData);
        } else {
            // Standard covariance: C = AA' (numPixels x numPixels)
            return centeredData.times(centeredData.transpose());
        }
    }

    private void sortEigenvectors(Matrix tempEigenValues, Matrix tempEigenVectors) {
        int n = tempEigenValues.getColumnDimension();

        // Create pairs for sorting
        double[] values = new double[n];
        int[] indices = new int[n];
        for (int i = 0; i < n; i++) {
            values[i] = tempEigenValues.get(i, i);
            indices[i] = i;
        }

        // Sort by eigenvalue (descending)
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (values[j] > values[i]) {
                    double tempVal = values[i];
                    values[i] = values[j];
                    values[j] = tempVal;

                    int tempIdx = indices[i];
                    indices[i] = indices[j];
                    indices[j] = tempIdx;
                }
            }
        }

        // Create sorted matrices
        eigenValues = new Matrix(n, n);
        eigenVectors = new Matrix(tempEigenVectors.getRowDimension(), n);

        for (int i = 0; i < n; i++) {
            eigenValues.set(i, i, values[i]);
            for (int j = 0; j < tempEigenVectors.getRowDimension(); j++) {
                eigenVectors.set(j, i, tempEigenVectors.get(j, indices[i]));
            }
        }
    }

    private void projectEigenvectorsToOriginalSpace(Matrix centeredData) {
        // u_i = A * v_i (project from sample space to pixel space)
        eigenVectors = centeredData.times(eigenVectors);
    }

    private void normalizeAndSelectComponents() {
        int totalComponents = eigenVectors.getColumnDimension();

        // Compute total variance
        double totalVariance = 0;
        for (int i = 0; i < eigenValues.getColumnDimension(); i++) {
            if (eigenValues.get(i, i) > EIGENVALUE_THRESHOLD) {
                totalVariance += eigenValues.get(i, i);
            }
        }

        // Normalize eigenvectors and count valid components
        numEigenVectors = 0;
        List<Double> varianceRatios = new ArrayList<>();

        for (int i = 0; i < totalComponents; i++) {
            double eigenValue = eigenValues.get(i, i);

            if (eigenValue > EIGENVALUE_THRESHOLD) {
                // Normalize this eigenvector
                double norm = 0;
                for (int j = 0; j < eigenVectors.getRowDimension(); j++) {
                    norm += eigenVectors.get(j, i) * eigenVectors.get(j, i);
                }
                norm = Math.sqrt(norm);

                if (norm > 0) {
                    for (int j = 0; j < eigenVectors.getRowDimension(); j++) {
                        eigenVectors.set(j, i, eigenVectors.get(j, i) / norm);
                    }
                }

                varianceRatios.add(eigenValue / totalVariance);
                numEigenVectors++;
            }
        }

        // Limit to requested number of components
        numEigenVectors = Math.min(numEigenVectors, config.getNumComponents());

        // Trim eigenVectors matrix
        eigenVectors = eigenVectors.getMatrix(0, eigenVectors.getRowDimension() - 1, 0, numEigenVectors - 1);

        // Store variance ratios
        explainedVarianceRatio = new double[numEigenVectors];
        for (int i = 0; i < numEigenVectors; i++) {
            explainedVarianceRatio[i] = varianceRatios.get(i);
        }
    }

    @Override
    public FeatureVector extract(FaceImage face) {
        if (!trained) {
            throw new IllegalStateException("Extractor not trained");
        }

        // Resize if necessary
        FaceImage resized = face.getWidth() != config.getImageWidth() ||
                           face.getHeight() != config.getImageHeight()
            ? face.resize(config.getImageWidth(), config.getImageHeight())
            : face;

        // Get grayscale pixels
        double[] pixels = resized.toGrayscaleArray();

        // Subtract mean face
        Matrix faceVector = new Matrix(pixels, pixels.length);
        Matrix centered = faceVector.minus(meanFace);

        // Project onto eigenfaces
        Matrix projection = eigenVectors.transpose().times(centered);

        // Extract coefficients
        double[] coefficients = new double[numEigenVectors];
        for (int i = 0; i < numEigenVectors; i++) {
            coefficients[i] = projection.get(i, 0);
        }

        // Optionally normalize
        if (config.isNormalize()) {
            double norm = 0;
            for (double c : coefficients) {
                norm += c * c;
            }
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
        return numEigenVectors;
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
        eigenVectors = null;
        eigenValues = null;
        numEigenVectors = 0;
        explainedVarianceRatio = null;
    }

    @Override
    public ExtractorConfig getConfig() {
        return config;
    }

    /**
     * Gets the mean face as a pixel array.
     *
     * @return the mean face pixels
     */
    public double[] getMeanFace() {
        if (!trained) {
            throw new IllegalStateException("Extractor not trained");
        }
        return meanFace.getColumnPackedCopy();
    }

    /**
     * Gets a specific eigenface as a pixel array.
     *
     * @param index the eigenface index (0-based)
     * @return the eigenface pixels
     */
    public double[] getEigenface(int index) {
        if (!trained) {
            throw new IllegalStateException("Extractor not trained");
        }
        if (index < 0 || index >= numEigenVectors) {
            throw new IndexOutOfBoundsException("Index out of range: " + index);
        }

        double[] eigenface = new double[eigenVectors.getRowDimension()];
        for (int i = 0; i < eigenface.length; i++) {
            eigenface[i] = eigenVectors.get(i, index);
        }
        return eigenface;
    }

    /**
     * Gets all eigenfaces.
     *
     * @return 2D array of eigenfaces [numEigenfaces][numPixels]
     */
    public double[][] getAllEigenfaces() {
        if (!trained) {
            throw new IllegalStateException("Extractor not trained");
        }

        double[][] eigenfaces = new double[numEigenVectors][eigenVectors.getRowDimension()];
        for (int i = 0; i < numEigenVectors; i++) {
            eigenfaces[i] = getEigenface(i);
        }
        return eigenfaces;
    }

    /**
     * Gets the explained variance ratio for each component.
     *
     * @return array of variance ratios
     */
    public double[] getExplainedVarianceRatio() {
        if (!trained) {
            throw new IllegalStateException("Extractor not trained");
        }
        return Arrays.copyOf(explainedVarianceRatio, explainedVarianceRatio.length);
    }

    /**
     * Gets the cumulative explained variance.
     *
     * @return the cumulative variance explained by all components
     */
    public double getCumulativeVariance() {
        if (!trained) {
            throw new IllegalStateException("Extractor not trained");
        }
        double sum = 0;
        for (double v : explainedVarianceRatio) {
            sum += v;
        }
        return sum;
    }

    /**
     * Reconstructs a face from its feature vector.
     *
     * @param features the feature vector
     * @return the reconstructed face pixels
     */
    public double[] reconstruct(FeatureVector features) {
        if (!trained) {
            throw new IllegalStateException("Extractor not trained");
        }

        double[] coefficients = features.getFeatures();
        Matrix projection = new Matrix(coefficients, coefficients.length);

        // Reconstruct: face = mean + V * w
        Matrix reconstructed = meanFace.plus(eigenVectors.times(projection));

        return reconstructed.getColumnPackedCopy();
    }

    @Override
    public String toString() {
        return String.format("EigenfacesExtractor{components=%d, trained=%s, variance=%.2f%%}",
            config.getNumComponents(), trained, trained ? getCumulativeVariance() * 100 : 0);
    }
}
