/* Author: Prasad U S
*
*
*/
package src;

class FeatureVector{
	private double[] 	featureVector;
	private int 		classification; 
	private Face 		face;
	
	
	public int getClassification() {
		return classification;
	}
	public void setClassification(int classification) {
		this.classification = classification;
	}
	public double[] getFeatureVector() {
		return featureVector;
	}
	public void setFeatureVector(double[] featureVector) {
		this.featureVector = featureVector;
	}
	public Face getFace() {
		return face;
	}
	public void setFace(Face face) {
		this.face = face;
	}

}
