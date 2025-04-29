/**
 * Manages the feature space for face recognition.
 * Handles feature vector storage, distance calculations, and classification.
 * Provides methods for k-nearest neighbor classification and feature space visualization.
 *
 * @author Prasad Subrahmanya
 * @version 1.0
 * @since 1.0
 */
package src;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public class FeatureSpace {

    /**
     * Euclidean distance measure implementation.
     * Calculates the distance between two feature vectors using the Euclidean metric.
     */
    public static final DistanceMeasure EUCLIDEAN_DISTANCE = new DistanceMeasure() {
        @Override
        public double calculateDistance(FeatureVector fv1, FeatureVector fv2) {
            double sum = 0;
            for (int i = 0; i < fv1.getFeatureVector().length; i++) {
                double diff = fv1.getFeatureVector()[i] - fv2.getFeatureVector()[i];
                sum += diff * diff;
            }
            return Math.sqrt(sum);
        }
    };
    private final ArrayList<FeatureVector> featureSpace;
    private final ArrayList<String> classifications;

    /**
     * Creates a new FeatureSpace instance.
     * Initializes empty collections for feature vectors and classifications.
     */
    public FeatureSpace() {
        featureSpace = new ArrayList<>();
        classifications = new ArrayList<>();
    }

    /**
     * Inserts a face and its feature vector into the database.
     * Associates the face with its classification and stores the feature vector.
     *
     * @param face the face to insert
     * @param featureVector the extracted feature vector for the face
     */
    public void insertIntoDatabase(Face face, double[] featureVector) {
        if (!classifications.contains(face.getClassification())) {
            classifications.add(face.getClassification());
        }
        int clas = classifications.indexOf(face.getClassification());

        FeatureVector obj = new FeatureVector();
        obj.setClassification(clas);
        obj.setFace(face);
        obj.setFeatureVector(featureVector);

        featureSpace.add(obj);
    }

    /**
     * Finds the classification of the closest feature vector.
     * Uses the specified distance measure to find the nearest neighbor.
     *
     * @param measure the distance measure to use
     * @param obj the feature vector to classify
     * @return the classification of the closest feature vector, or null if the feature space is empty
     */
    public String closestFeature(DistanceMeasure measure, FeatureVector obj) {
        if (getFeatureSpaceSize() < 1) {
            return null;
        }

        String result = classifications.get(featureSpace.get(0).getClassification());
        double distance = measure.calculateDistance(obj, featureSpace.get(0));
        for (int i = 1; i < featureSpace.size(); i++) {
            double currentDistance = measure.calculateDistance(obj, featureSpace.get(i));
            if (currentDistance < distance) {
                distance = currentDistance;
                result = classifications.get(featureSpace.get(i).getClassification());
            }
        }
        return result;
    }

    /**
     * Performs k-nearest neighbor classification.
     * Uses the specified distance measure to find the k nearest neighbors.
     *
     * @param measure the distance measure to use
     * @param fv the feature vector to classify
     * @param k the number of neighbors to consider
     * @return the classification of the majority of k nearest neighbors, or null if the feature space is empty
     */
    public String knn(DistanceMeasure measure, FeatureVector fv, int k) {
        FaceDistancePair[] distances = orderByDistance(measure, fv);
        if (distances.length == 0) {
            return null;
        }
        return distances[0].getFace().getClassification();
    }

    /**
     * Orders all faces by their distance to a probe feature vector.
     * Creates an array of face-distance pairs sorted by increasing distance.
     *
     * @param measure the distance measure to use
     * @param fv the probe feature vector
     * @return sorted array of face-distance pairs
     */
    public FaceDistancePair[] orderByDistance(DistanceMeasure measure, FeatureVector fv) {
        FaceDistancePair[] distances = new FaceDistancePair[featureSpace.size()];
        for (int i = 0; i < featureSpace.size(); i++) {
            distances[i] = new FaceDistancePair();
            distances[i].setFace(featureSpace.get(i).getFace());
            distances[i].setDist(measure.calculateDistance(fv, featureSpace.get(i)));
        }
        Arrays.sort(distances, (a, b) -> Double.compare(a.getDist(), b.getDist()));
        return distances;
    }

    /**
     * Inner class representing a face and its distance to a probe vector.
     * Used for sorting and displaying face matches by similarity.
     */
    public static class FaceDistancePair {
        private Face face;
        private double dist;

        /**
         * Gets the face in this pair.
         *
         * @return the face
         */
        public Face getFace() {
            return face;
        }

        /**
         * Sets the face in this pair.
         *
         * @param face the face to set
         */
        public void setFace(Face face) {
            this.face = face;
        }

        /**
         * Gets the distance value in this pair.
         *
         * @return the distance value
         */
        public double getDist() {
            return dist;
        }

        /**
         * Sets the distance value in this pair.
         *
         * @param dist the distance value to set
         */
        public void setDist(double dist) {
            this.dist = dist;
        }
    }

    /**
     * Gets a 3D representation of the feature space.
     * Creates a normalized array of 3D points representing feature vectors.
     *
     * @return array of 3D points representing the feature space
     */
    public double[][] get3dFeatureSpace() {
        double[][] features = new double[classifications.size() * 18 + 18][3];
        for (int i = 0; i < classifications.size(); i++) {

            ArrayList<FeatureVector> rightClass = new ArrayList<FeatureVector>();
            for (int j = 0; j < featureSpace.size(); j++) {
                if (featureSpace.get(j).getClassification() == i) {
                    rightClass.add(featureSpace.get(j));
                }
            }

            for (int j = 0; j < 18; j++) {
                int pos = i * 18 + j;
                int tmp = j % rightClass.size();
                features[pos][0] = rightClass.get(tmp).getFeatureVector()[0];
                features[pos][1] = rightClass.get(tmp).getFeatureVector()[1];
                features[pos][2] = rightClass.get(tmp).getFeatureVector()[2];
            }
        }


        double max0 = features[0][0], max1 = features[0][1], max2 = features[0][2];
        double min0 = features[0][0], min1 = features[0][1], min2 = features[0][2];
        for (int i = 1; i < features.length - 18; i++) {                       
            // get the max and min on each axis
            if (features[i][0] > max0) {
                max0 = features[i][0];
            }
            if (features[i][0] < min0) {
                min0 = features[i][0];
            }

            if (features[i][1] > max1) {
                max1 = features[i][1];
            }
            if (features[i][1] < min1) {
                min1 = features[i][1];
            }

            if (features[i][2] > max2) {
                max2 = features[i][2];
            }
            if (features[i][2] < min2) {
                min2 = features[i][2];
            }
        }

        double mult0 = (max0 - min0) / 100;
        double mult1 = (max1 - min1) / 100;
        double mult2 = (max2 - min2) / 100;

        for (int i = 0; i < features.length - 18; i++) {                       
            // perform the normalisation
            features[i][0] -= min0;
            features[i][0] /= mult0;

            features[i][1] -= min1;
            features[i][1] /= mult1;

            features[i][2] -= min2;
            features[i][2] /= mult2;
        }

        return features;
    }

    /**
     * Gets a 3D representation of the feature space including a probe vector.
     * Creates a normalized array of 3D points with the probe vector highlighted.
     *
     * @param probe the probe feature vector to include
     * @return array of 3D points representing the feature space with probe
     */
    public double[][] get3dFeatureSpace(FeatureVector probe) {
        if (probe == null) {
            return get3dFeatureSpace();
        }
        double[][] features = new double[classifications.size() * 18 + 36][3];
        for (int i = 0; i < classifications.size(); i++) {

            ArrayList<FeatureVector> rightClass = new ArrayList<FeatureVector>();
            for (int j = 0; j < featureSpace.size(); j++) {
                if (featureSpace.get(j).getClassification() == i) {
                    rightClass.add(featureSpace.get(j));
                }
            }


            for (int j = 0; j < 18; j++) {
                int pos = i * 18 + j;
                int tmp = j % rightClass.size();
                features[pos][0] = rightClass.get(tmp).getFeatureVector()[0];
                features[pos][1] = rightClass.get(tmp).getFeatureVector()[1];
                features[pos][2] = rightClass.get(tmp).getFeatureVector()[2];
            }
        }

        for (int j = 0; j < 18; j++) {
            int pos = featureSpace.size() + j;
            features[pos][0] = probe.getFeatureVector()[0];
            features[pos][1] = probe.getFeatureVector()[1];
            features[pos][2] = probe.getFeatureVector()[2];
        }

        //norlamise these values from 0-100
        double max0 = features[0][0], max1 = features[0][1], max2 = features[0][2];
        double min0 = features[0][0], min1 = features[0][1], min2 = features[0][2];
        for (int i = 1; i < features.length - 18; i++) {                       //get the max and min on each axis
            if (features[i][0] > max0) {
                max0 = features[i][0];
            }
            if (features[i][0] < min0) {
                min0 = features[i][0];
            }

            if (features[i][1] > max1) {
                max1 = features[i][1];
            }
            if (features[i][1] < min1) {
                min1 = features[i][1];
            }

            if (features[i][2] > max2) {
                max2 = features[i][2];
            }
            if (features[i][2] < min2) {
                min2 = features[i][2];
            }
        }

        double mult0 = (max0 - min0) / 100;
        double mult1 = (max1 - min1) / 100;
        double mult2 = (max2 - min2) / 100;

        for (int i = 0; i < features.length - 18; i++) {
            features[i][0] -= min0;
            features[i][0] /= mult0;

            features[i][1] -= min1;
            features[i][1] /= mult1;

            features[i][2] -= min2;
            features[i][2] /= mult2;
        }

        return features;
    }

    /**
     * Gets the number of feature vectors in the feature space.
     *
     * @return the size of the feature space
     */
    public int getFeatureSpaceSize() {
        return featureSpace.size();
    }
}
