package src;

/**
 * Interface for calculating distance between two feature vectors.
 * Different implementations can provide different distance metrics
 * (e.g., Euclidean, Manhattan, etc.).
 *
 * @author Prasad Subrahmanya
 * @version 1.0
 */
public interface DistanceMeasure {
	/**
	 * Calculates the distance between two feature vectors.
	 *
	 * @param vector1 the first feature vector
	 * @param vector2 the second feature vector
	 * @return the calculated distance between the vectors
	 */
	double calculateDistance(FeatureVector vector1, FeatureVector vector2);
}
