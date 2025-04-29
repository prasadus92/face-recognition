/**
 * Two-Stage Classification and Detection (TSCD) implementation for face recognition.
 * This class handles the core face recognition algorithms using eigenfaces.
 *
 * @author Prasad Subrahmanya
 * @version 1.0
 * @since 1.0
 */
package src;

import java.util.Arrays;
import java.util.Comparator;
import src.Main.ProgressTracker;
import Jama.EigenvalueDecomposition;
import Jama.Matrix;

public class TSCD {
    private static final double EIGENVALUE_THRESHOLD = 0.0001;
    private static final int DEFAULT_EIGEN_VECTORS = 10;

    private Matrix averageFace;        // Stores the average face useful when probing the database
    private Matrix eigenVectors;       // Stores all the sorted eigen vectors from the training set
    private Matrix eigenValues;        // Stores all the sorted eigen Values from the training set
    private boolean trained;           // Has a training set been provided yet?
    private int numEigenVectors;       // Number of eigen vectors available

    /**
     * Processes the training set of faces to compute eigenfaces.
     *
     * @param faces Array of face images to process
     * @param progress Progress tracker for monitoring the process
     */
    public void processTrainingSet(Face[] faces, ProgressTracker progress) {
        progress.advanceProgress("Constructing matrix...");
        double[][] dpix = constructImageMatrix(faces);

        progress.advanceProgress("Calculating averages...");
        computeAverageFace(dpix);

        progress.advanceProgress("Computing covariance matrix...");
        Matrix A = computeCovarianceMatrix(dpix);

        progress.advanceProgress("Calculating eigenvectors...");
        computeEigenvectors(A);

        progress.advanceProgress("Sorting eigenvectors...");
        sortEigenvectors();

        progress.advanceProgress("Extracting eigenvalues...");
        extractEigenvalues(A);

        progress.advanceProgress("Normalising eigenvectors...");
        normalizeEigenvectors();

        trained = true;
        progress.finished();
    }

    /**
     * Constructs a matrix from the face images.
     *
     * @param faces Array of face images
     * @return Matrix containing pixel values from all faces
     */
    private double[][] constructImageMatrix(Face[] faces) {
        double[][] dpix = new double[faces.length][faces[0].getPicture().getImagePixels().length];
        for (int i = 0; i < faces.length; i++) {
            double[] pixels = faces[i].getPicture().getImagePixels();
            System.arraycopy(pixels, 0, dpix[i], 0, pixels.length);
        }
        return dpix;
    }

    /**
     * Computes the average face from the training set.
     *
     * @param dpix Matrix containing pixel values from all faces
     */
    private void computeAverageFace(double[][] dpix) {
        Matrix matrix = new Matrix(dpix);
        averageFace = new Matrix(1, matrix.getColumnDimension());
        for (int i = 0; i < matrix.getRowDimension(); i++) {
            averageFace.plusEquals(matrix.getMatrix(i, i, 0, matrix.getColumnDimension() - 1));
        }
        averageFace.timesEquals(1.0 / matrix.getRowDimension());
    }

    /**
     * Computes the covariance matrix for the face images.
     *
     * @param dpix Matrix containing pixel values from all faces
     * @return Transposed difference matrix
     */
    private Matrix computeCovarianceMatrix(double[][] dpix) {
        Matrix matrix = new Matrix(dpix);
        Matrix bigAvg = new Matrix(matrix.getRowDimension(), matrix.getColumnDimension());
        for (int i = 0; i < bigAvg.getRowDimension(); i++) {
            bigAvg.setMatrix(i, i, 0, bigAvg.getColumnDimension() - 1, averageFace);
        }
        return matrix.minus(bigAvg).transpose();
    }

    /**
     * Computes eigenvectors and eigenvalues from the covariance matrix.
     *
     * @param A Transposed difference matrix
     */
    private void computeEigenvectors(Matrix A) {
        Matrix At = A.transpose();
        Matrix L = At.times(A);
        EigenvalueDecomposition eigen = L.eig();
        eigenValues = eigen.getD();
        eigenVectors = eigen.getV();
    }

    /**
     * Sorts eigenvectors based on their corresponding eigenvalues.
     */
    private void sortEigenvectors() {
        Matrix[] eigDVSorted = sortem(eigenValues, eigenVectors);
        eigenValues = eigDVSorted[0];
        eigenVectors = eigDVSorted[1];
    }

    /**
     * Extracts and normalizes eigenvalues.
     *
     * @param A Transposed difference matrix
     */
    private void extractEigenvalues(Matrix A) {
        double[] values = diag(eigenValues);
        for (int i = 0; i < values.length; i++) {
            values[i] /= A.getColumnDimension() - 1;
        }
    }

    /**
     * Normalizes eigenvectors and removes those with small eigenvalues.
     */
    private void normalizeEigenvectors() {
        numEigenVectors = 0;
        for (int i = 0; i < eigenVectors.getColumnDimension(); i++) {
            Matrix tmp;
            if (eigenValues.get(i, i) < EIGENVALUE_THRESHOLD) {
                tmp = new Matrix(eigenVectors.getRowDimension(), 1);
            } else {
                tmp = eigenVectors.getMatrix(0, eigenVectors.getRowDimension() - 1, i, i)
                    .times(1 / eigenVectors.getMatrix(0, eigenVectors.getRowDimension() - 1, i, i).normF());
                numEigenVectors++;
            }
            eigenVectors.setMatrix(0, eigenVectors.getRowDimension() - 1, i, i, tmp);
        }
        eigenVectors = eigenVectors.getMatrix(0, eigenVectors.getRowDimension() - 1, 0, numEigenVectors - 1);
    }

    /**
     * Gets the eigenface representation of a picture.
     *
     * @param pic The picture to process
     * @param number Number of eigenfaces to use
     * @return Array of eigenface coefficients
     */
    public double[] getEigenFaces(Picture pic, int number) {
        if (number > numEigenVectors) {
            number = numEigenVectors;
        }

        double[] pixels = pic.getImagePixels();
        Matrix face = new Matrix(pixels, pixels.length);
        Matrix vectors = eigenVectors.getMatrix(0, eigenVectors.getRowDimension() - 1, 0, number - 1).transpose();
        Matrix result = vectors.times(face);

        double[] coefficients = new double[number];
        for (int i = 0; i < number; i++) {
            coefficients[i] = result.get(i, 0);
        }
        return coefficients;
    }

    /**
     * Extracts diagonal elements from a matrix.
     *
     * @param matrix Input matrix
     * @return Array of diagonal elements
     */
    private double[] diag(Matrix matrix) {
        double[] diagonal = new double[matrix.getColumnDimension()];
        for (int i = 0; i < matrix.getColumnDimension(); i++) {
            diagonal[i] = matrix.get(i, i);
        }
        return diagonal;
    }

    /**
     * Sorts eigenvectors and eigenvalues.
     *
     * @param eigenvalues Matrix of eigenvalues
     * @param eigenvectors Matrix of eigenvectors
     * @return Array containing sorted eigenvalues and eigenvectors
     */
    private Matrix[] sortem(Matrix eigenvalues, Matrix eigenvectors) {
        double[] values = diag(eigenvalues);
        EigenPair[] pairs = new EigenPair[values.length];
        
        for (int i = 0; i < pairs.length; i++) {
            pairs[i] = new EigenPair();
            pairs[i].index = i;
            pairs[i].value = values[i];
        }

        Arrays.sort(pairs, (a, b) -> Double.compare(b.value, a.value));

        Matrix sortedEigenvalues = new Matrix(eigenvalues.getRowDimension(), eigenvalues.getColumnDimension());
        Matrix sortedEigenvectors = new Matrix(eigenvectors.getRowDimension(), eigenvectors.getColumnDimension());

        for (int i = 0; i < pairs.length; i++) {
            sortedEigenvalues.set(i, i, pairs[i].value);
            sortedEigenvectors.setMatrix(0, sortedEigenvectors.getRowDimension() - 1, i, i,
                eigenvectors.getMatrix(0, eigenvectors.getRowDimension() - 1, pairs[i].index, pairs[i].index));
        }

        return new Matrix[]{sortedEigenvalues, sortedEigenvectors};
    }

    /**
     * Checks if the system has been trained.
     *
     * @return true if trained, false otherwise
     */
    public boolean isTrained() {
        return trained;
    }

    /**
     * Gets the number of available eigenvectors.
     *
     * @return Number of eigenvectors
     */
    public int getNumEigenVectors() {
        return numEigenVectors;
    }

    /**
     * Gets the average face matrix.
     *
     * @return The average face matrix
     */
    public Matrix getAverageFace() {
        return averageFace;
    }

    /**
     * Helper class for sorting eigenvalues and eigenvectors.
     */
    private static class EigenPair {
        double value;
        int index;
    }
}
