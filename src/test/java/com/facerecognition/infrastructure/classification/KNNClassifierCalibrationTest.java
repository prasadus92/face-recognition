package com.facerecognition.infrastructure.classification;

import com.facerecognition.domain.model.FeatureVector;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.domain.service.FaceClassifier;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.offset;

@DisplayName("KNNClassifier confidence calibration")
class KNNClassifierCalibrationTest {

    @Test
    @DisplayName("a probe near one person's photos is confident; a stranger between people is not")
    void separatesMatchesFromStrangers() {
        KNNClassifier classifier = galleryOf(person("ann", 0, 0), person("bo", 10, 0));

        assertThat(classifier.classify(vector(0.2, 0.1), 0.0).getConfidence()).isGreaterThan(0.9);
        assertThat(classifier.classify(vector(5, 8), 0.0).getConfidence()).isLessThan(0.1);
        assertThat(classifier.classify(vector(5, 8), 0.6).isRecognized()).isFalse();
    }

    @Test
    @DisplayName("confidence does not depend on the units of the distance")
    void isScaleFree() {
        double small = galleryOf(person("ann", 0, 0), person("bo", 10, 0))
                .classify(vector(1, 0), 0.0).getConfidence();
        double large = galleryOf(person("ann", 0, 0, 1000), person("bo", 10_000, 0, 1000))
                .classify(vector(1000, 0), 0.0).getConfidence();

        assertThat(large).isCloseTo(small, offset(1e-9));
    }

    @Test
    @DisplayName("enrolling another person recalibrates")
    void recalibratesOnEnroll() {
        KNNClassifier classifier = galleryOf(person("ann", 0, 0), person("bo", 10, 0));
        RecognitionResult before = classifier.classify(vector(1.5, 0), 0.0);

        classifier.enroll(person("cy", 4, 0));
        RecognitionResult after = classifier.classify(vector(1.5, 0), 0.0);

        assertThat(after.getBestMatch().orElseThrow().getIdentity().getName()).isEqualTo("ann");
        assertThat(after.getBestMatch().orElseThrow().getDistance())
                .isEqualTo(before.getBestMatch().orElseThrow().getDistance());
        assertThat(after.getConfidence()).isLessThan(before.getConfidence());
    }

    @Test
    @DisplayName("with 1 photo per person a stranger halfway between people is rejected")
    void onePhotoPerPerson() {
        KNNClassifier classifier = galleryOf(single("ann", 0, 0), single("bo", 10, 0));

        assertThat(classifier.classify(vector(5, 0), 0.6).isRecognized()).isFalse();
        assertThat(classifier.classify(vector(0.5, 0), 0.0).getConfidence()).isGreaterThan(0.9);
    }

    @Test
    @DisplayName("with 1 person there is nothing to calibrate against and the fixed scale applies")
    void singlePersonFallsBack() {
        KNNClassifier classifier = galleryOf(single("ann", 0, 0));

        assertThat(classifier.classify(vector(3, 4), 0.0).getConfidence()).isCloseTo(Math.exp(-5 / 5000.0), offset(1e-12));
    }

    @Test
    @DisplayName("inactive people and unenrolled people do not take part in calibration")
    void ignoresInactiveAndRemovedPeople() {
        double reference = galleryOf(person("ann", 0, 0), person("bo", 10, 0)).classify(vector(1.5, 0), 0.0).getConfidence();

        Identity inactive = person("cy", 3, 0);
        inactive.setActive(false);
        assertThat(galleryOf(person("ann", 0, 0), person("bo", 10, 0), inactive)
                .classify(vector(1.5, 0), 0.0).getConfidence()).isEqualTo(reference);

        Identity removed = person("dee", 3, 0);
        KNNClassifier classifier = galleryOf(person("ann", 0, 0), person("bo", 10, 0), removed);
        classifier.unenroll(removed.getId());
        assertThat(classifier.classify(vector(1.5, 0), 0.0).getConfidence()).isEqualTo(reference);
    }

    @Test
    @DisplayName("in average mode the genuine distance is measured to the person's average")
    void averageModeUsesAverages() {
        KNNClassifier classifier = new KNNClassifier(new FaceClassifier.ClassifierConfig().setUseAverageFeatures(true));
        classifier.enroll(triangle("ann", 0));
        classifier.enroll(triangle("bo", 10));
        classifier.retrain();

        // Photo-to-average distances are 0.471, 0.745 and 0.745 per person, so the genuine median is
        // 0.745, the distance from (1, 0) to ann's average; the curve gives 1/(1+e^-4) there.
        assertThat(classifier.classify(vector(1, 0), 0.0).getConfidence()).isCloseTo(1 / (1 + Math.exp(-4)), offset(1e-9));
    }

    private static Identity triangle(String name, double x) {
        Identity identity = new Identity(name);
        identity.enrollSample(vector(x, 0), 1.0, "test");
        identity.enrollSample(vector(x + 1, 0), 1.0, "test");
        identity.enrollSample(vector(x, 1), 1.0, "test");
        return identity;
    }

    private static Identity single(String name, double x, double y) {
        Identity identity = new Identity(name);
        identity.enrollSample(vector(x, y), 1.0, "test");
        return identity;
    }

    private static KNNClassifier galleryOf(Identity... people) {
        KNNClassifier classifier = new KNNClassifier();
        for (Identity person : people) {
            classifier.enroll(person);
        }
        classifier.retrain();
        return classifier;
    }

    private static Identity person(String name, double x, double y) {
        return person(name, x, y, 1);
    }

    /** 3 photos around (x, y), spread by {@code unit / 2}. */
    private static Identity person(String name, double x, double y, double unit) {
        Identity identity = new Identity(name);
        identity.enrollSample(vector(x, y), 1.0, "test");
        identity.enrollSample(vector(x + unit / 2, y), 1.0, "test");
        identity.enrollSample(vector(x, y + unit / 2), 1.0, "test");
        return identity;
    }

    private static FeatureVector vector(double x, double y) {
        return new FeatureVector(new double[]{x, y}, "test", 1);
    }
}
