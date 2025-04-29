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
    private Matrix eigVectors;         // Stores all the sorted eigen vectors from the training set
    private Matrix eigValues;          // Stores all the sorted eigen Values from the training set
    private boolean trained;           // Has a training set been provided yet?
    private int numEigenVecs;          // Number of eigen vectors available

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
        double[][] dpix = new double[faces.length][faces[0].picture.getImagePixels().length];
        for (int i = 0; i < faces.length; i++) {
            double[] pixels = faces[i].picture.getImagePixels();
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
        eigValues = eigen.getD();
        eigVectors = eigen.getV();
    }

    /**
     * Sorts eigenvectors based on their corresponding eigenvalues.
     */
    private void sortEigenvectors() {
        Matrix[] eigDVSorted = sortem(eigValues, eigVectors);
        eigValues = eigDVSorted[0];
        eigVectors = eigDVSorted[1];
    }

    /**
     * Extracts and normalizes eigenvalues.
     *
     * @param A Transposed difference matrix
     */
    private void extractEigenvalues(Matrix A) {
        double[] values = diag(eigValues);
        for (int i = 0; i < values.length; i++) {
            values[i] /= A.getColumnDimension() - 1;
        }
    }

    /**
     * Normalizes eigenvectors and removes those with small eigenvalues.
     */
    private void normalizeEigenvectors() {
        numEigenVecs = 0;
        for (int i = 0; i < eigVectors.getColumnDimension(); i++) {
            Matrix tmp;
            if (eigValues.get(i, i) < EIGENVALUE_THRESHOLD) {
                tmp = new Matrix(eigVectors.getRowDimension(), 1);
            } else {
                tmp = eigVectors.getMatrix(0, eigVectors.getRowDimension() - 1, i, i)
                    .times(1 / eigVectors.getMatrix(0, eigVectors.getRowDimension() - 1, i, i).normF());
                numEigenVecs++;
            }
            eigVectors.setMatrix(0, eigVectors.getRowDimension() - 1, i, i, tmp);
        }
        eigVectors = eigVectors.getMatrix(0, eigVectors.getRowDimension() - 1, 0, numEigenVecs - 1);
    }

    /**
     * Gets the eigenface representation of a picture.
     *
     * @param pic The picture to process
     * @param number Number of eigenfaces to use
     * @return Array of eigenface coefficients
     */
    public double[] getEigenFaces(Picture pic, int number) {
        if (number > numEigenVecs) {
            number = numEigenVecs;
        }

        double[] pixels = pic.getImagePixels();
        Matrix face = new Matrix(pixels, pixels.length);
        Matrix Vecs = eigVectors.getMatrix(0, eigVectors.getRowDimension() - 1, 0, number - 1).transpose();
        Matrix rslt = Vecs.times(face);

        double[] ret = new double[number];
        for (int i = 0; i < number; i++) {
            ret[i] = rslt.get(i, 0);
        }
        return ret;
    }

    /**
     * Extracts diagonal elements from a matrix.
     *
     * @param M Input matrix
     * @return Array of diagonal elements
     */
    private double[] diag(Matrix M) {
        double[] dvec = new double[M.getColumnDimension()];
        for (int i = 0; i < M.getColumnDimension(); i++) {
            dvec[i] = M.get(i, i);
        }
        return dvec;
    }

    /**
     * Sorts eigenvectors and eigenvalues.
     *
     * @param D Matrix of eigenvalues
     * @param V Matrix of eigenvectors
     * @return Array containing sorted eigenvalues and eigenvectors
     */
    private Matrix[] sortem(Matrix D, Matrix V) {
        double[] dvec = diag(D);
        di_pair[] dvec_indexed = new di_pair[dvec.length];
        
        for (int i = 0; i < dvec_indexed.length; i++) {
            dvec_indexed[i] = new di_pair();
            dvec_indexed[i].index = i;
            dvec_indexed[i].value = dvec[i];
        }

        Arrays.sort(dvec_indexed, (arg0, arg1) -> {
            di_pair lt = (di_pair) arg0;
            di_pair rt = (di_pair) arg1;
            double dif = (lt.value - rt.value);
            return dif > 0 ? -1 : (dif < 0 ? 1 : 0);
        });

        Matrix D2 = new Matrix(D.getRowDimension(), D.getColumnDimension());
        Matrix V2 = new Matrix(V.getRowDimension(), V.getColumnDimension());

        for (int i = 0; i < dvec_indexed.length; i++) {
            D2.set(i, i, D.get(dvec_indexed[i].index, dvec_indexed[i].index));
            int height = V.getRowDimension() - 1;
            Matrix tmp = V.getMatrix(dvec_indexed[i].index, dvec_indexed[i].index, 0, height);
            V2.setMatrix(i, i, 0, height, tmp);
        }

        Matrix V3 = new Matrix(V.getRowDimension(), V.getColumnDimension());
        for (int i = 0; i < V3.getRowDimension(); i++) {
            for (int j = 0; j < V3.getColumnDimension(); j++) {
                V3.set(i, j, V2.get(V3.getRowDimension() - i - 1, V3.getColumnDimension() - j - 1));
            }
        }

        return new Matrix[]{D2, V3};
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
    public int getNumEigenVecs() {
        return numEigenVecs;
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
    private static class di_pair {
        double value;
        int index;
    }
}
