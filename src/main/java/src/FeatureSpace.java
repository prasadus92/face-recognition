/* Author: Prasad U S
*
*
*/
package src;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;

public class FeatureSpace {

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
    private ArrayList<FeatureVector> featureSpace;
    private ArrayList<String> classifications;

    public FeatureSpace() {
        featureSpace = new ArrayList<FeatureVector>();
        classifications = new ArrayList<String>();
    }

    public void insertIntoDatabase(Face face, double[] featureVector) {
        if (!classifications.contains(face.classification)) {
            classifications.add(face.classification);
        }
        int clas = classifications.indexOf(face.classification);

        FeatureVector obj = new FeatureVector();
        obj.setClassification(clas);
        obj.setFace(face);
        obj.setFeatureVector(featureVector);

        featureSpace.add(obj);
    }

    public String closestFeature(DistanceMeasure measure, FeatureVector obj) {
        if (getFeatureSpaceSize() < 1) {
            return null;
        }

        String ret = classifications.get(featureSpace.get(0).getClassification());
        double dist = measure.calculateDistance(obj, featureSpace.get(0));
        for (int i = 1; i < featureSpace.size(); i++) {
            double d = measure.calculateDistance(obj, featureSpace.get(i));
            if (d < dist) {
                dist = d;
                ret = classifications.get(featureSpace.get(i).getClassification());
            }
        }
        return ret;
    }

    public String knn(DistanceMeasure measure, FeatureVector fv, int k) {
        FeatureSpace.fd_pair[] distances = orderByDistance(measure, fv);
        if (distances.length == 0) {
            return null;
        }
        return distances[0].face.getClassification();
    }

    public FeatureSpace.fd_pair[] orderByDistance(DistanceMeasure measure, FeatureVector fv) {
        FeatureSpace.fd_pair[] distances = new FeatureSpace.fd_pair[featureSpace.size()];
        for (int i = 0; i < featureSpace.size(); i++) {
            distances[i] = new FeatureSpace.fd_pair();
            distances[i].face = featureSpace.get(i).getFace();
            distances[i].dist = measure.calculateDistance(fv, featureSpace.get(i));
        }
        Arrays.sort(distances, (a, b) -> Double.compare(a.dist, b.dist));
        return distances;
    }

    public class fd_pair {

        public Face face;
        public double dist;
    }

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

    public int getFeatureSpaceSize() {
        return featureSpace.size();
    }
}
