/* Author: Prasad U S
*
*
*/
package src;

/**
 * Represents a feature vector in the face recognition system.
 * A feature vector contains the extracted features from a face image
 * and its classification information.
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
	 *
	 * @return the classification value
	 */
	public int getClassification() {
		return classification;
	}
	
	/**
	 * Sets the classification of this feature vector.
	 *
	 * @param classification the classification value to set
	 */
	public void setClassification(int classification) {
		this.classification = classification;
	}
	
	/**
	 * Gets the feature vector array.
	 *
	 * @return the feature vector array
	 */
	public double[] getFeatureVector() {
		return featureVector;
	}
	
	/**
	 * Sets the feature vector array.
	 *
	 * @param featureVector the feature vector array to set
	 */
	public void setFeatureVector(double[] featureVector) {
		this.featureVector = featureVector;
	}
	
	/**
	 * Gets the associated face.
	 *
	 * @return the associated face
	 */
	public Face getFace() {
		return face;
	}
	
	/**
	 * Sets the associated face.
	 *
	 * @param face the face to associate with this feature vector
	 */
	public void setFace(Face face) {
		this.face = face;
	}
}
