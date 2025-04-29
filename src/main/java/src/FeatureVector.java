package src;

/**
 * Represents a feature vector in the face recognition system.
 * A feature vector contains the extracted features from a face image
 * and its classification information. It stores both the numerical
 * feature representation and associated metadata.
 *
 * @author Prasad Subrahmanya
 * @version 1.0
 * @since 1.0
 */
public class FeatureVector {
	private double[] featureVector;
	private int classification;
	private Face face;
	
	/**
	 * Gets the classification of this feature vector.
	 * The classification represents the identity or category 
	 * assigned to this feature vector.
	 *
	 * @return the classification value identifying this feature vector
	 */
	public int getClassification() {
		return classification;
	}
	
	/**
	 * Sets the classification of this feature vector.
	 * The classification should be a unique identifier representing
	 * the identity or category of the face.
	 *
	 * @param classification the classification value to set
	 */
	public void setClassification(int classification) {
		this.classification = classification;
	}
	
	/**
	 * Gets the feature vector array containing the extracted face features.
	 * These features are typically eigenface coefficients or other
	 * numerical representations of facial characteristics.
	 *
	 * @return the feature vector array containing face features
	 */
	public double[] getFeatureVector() {
		return featureVector;
	}
	
	/**
	 * Sets the feature vector array containing the extracted face features.
	 * The array should contain properly normalized feature values.
	 *
	 * @param featureVector the feature vector array to set
	 */
	public void setFeatureVector(double[] featureVector) {
		this.featureVector = featureVector;
	}
	
	/**
	 * Gets the associated face object.
	 * This provides access to the original face image and its properties.
	 *
	 * @return the associated face object
	 */
	public Face getFace() {
		return face;
	}
	
	/**
	 * Sets the associated face object.
	 * Links this feature vector to its source face image.
	 *
	 * @param face the face object to associate with this feature vector
	 */
	public void setFace(Face face) {
		this.face = face;
	}
}
