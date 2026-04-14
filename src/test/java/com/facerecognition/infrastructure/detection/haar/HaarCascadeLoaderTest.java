package com.facerecognition.infrastructure.detection.haar;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIOException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.nio.charset.StandardCharsets;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Sanity-check parser tests for {@link HaarCascadeLoader}. These don't try to
 * prove detection accuracy; they prove that we can load the bundled cascade
 * and a minimal hand-rolled one without losing data.
 */
@DisplayName("HaarCascadeLoader")
class HaarCascadeLoaderTest {

    @Test
    @DisplayName("parses the bundled OpenCV frontal-face cascade into a non-trivial model")
    void parsesBundledCascade() throws IOException {
        HaarCascade cascade = HaarCascadeLoader.loadFromClasspath(
                "cascades/haarcascade_frontalface_default.xml");

        // OpenCV's default frontal-face cascade is a 24x24 window, 25 stages,
        // with thousands of features. If any of these numbers drift unexpectedly
        // we want to know — but allow some tolerance in case we swap cascade
        // files later.
        assertThat(cascade.getWindowWidth()).isEqualTo(24);
        assertThat(cascade.getWindowHeight()).isEqualTo(24);
        assertThat(cascade.getStageCount()).isBetween(20, 30);
        assertThat(cascade.getFeatureCount()).isGreaterThan(1000);

        // Every weak classifier must reference a valid feature.
        for (HaarCascade.Stage stage : cascade.getStages()) {
            assertThat(stage.getClassifiers()).isNotEmpty();
            for (HaarCascade.WeakClassifier weak : stage.getClassifiers()) {
                assertThat(weak.getFeatureIndex())
                        .isBetween(0, cascade.getFeatureCount() - 1);
                HaarCascade.Feature feature = cascade.getFeatures()[weak.getFeatureIndex()];
                assertThat(feature.getRects()).hasSizeBetween(2, 3);
                for (HaarCascade.Rect rect : feature.getRects()) {
                    assertThat(rect.getWidth()).isPositive();
                    assertThat(rect.getHeight()).isPositive();
                }
            }
        }
    }

    @Test
    @DisplayName("parses a minimal one-stage hand-rolled cascade")
    void parsesMinimalInlineCascade() throws IOException {
        String xml = ""
                + "<?xml version=\"1.0\"?>"
                + "<opencv_storage>"
                + "  <cascade type_id=\"opencv-cascade-classifier\">"
                + "    <stageType>BOOST</stageType>"
                + "    <featureType>HAAR</featureType>"
                + "    <height>12</height>"
                + "    <width>12</width>"
                + "    <stageNum>1</stageNum>"
                + "    <stages>"
                + "      <_>"
                + "        <maxWeakCount>1</maxWeakCount>"
                + "        <stageThreshold>-0.5</stageThreshold>"
                + "        <weakClassifiers>"
                + "          <_>"
                + "            <internalNodes>0 -1 0 1.25e-02</internalNodes>"
                + "            <leafValues>-0.75 0.75</leafValues>"
                + "          </_>"
                + "        </weakClassifiers>"
                + "      </_>"
                + "    </stages>"
                + "    <features>"
                + "      <_>"
                + "        <rects>"
                + "          <_>0 0 12 6 -1.</_>"
                + "          <_>0 6 12 6 1.</_>"
                + "        </rects>"
                + "        <tilted>0</tilted>"
                + "      </_>"
                + "    </features>"
                + "  </cascade>"
                + "</opencv_storage>";

        HaarCascade cascade = HaarCascadeLoader.load(
                new ByteArrayInputStream(xml.getBytes(StandardCharsets.UTF_8)));

        assertThat(cascade.getWindowWidth()).isEqualTo(12);
        assertThat(cascade.getWindowHeight()).isEqualTo(12);
        assertThat(cascade.getStageCount()).isEqualTo(1);
        assertThat(cascade.getFeatureCount()).isEqualTo(1);

        HaarCascade.Stage stage = cascade.getStages()[0];
        assertThat(stage.getThreshold()).isEqualTo(-0.5f);
        assertThat(stage.getClassifiers()).hasSize(1);

        HaarCascade.WeakClassifier weak = stage.getClassifiers()[0];
        assertThat(weak.getFeatureIndex()).isZero();
        assertThat(weak.getLeftLeaf()).isEqualTo(-0.75f);
        assertThat(weak.getRightLeaf()).isEqualTo(0.75f);

        HaarCascade.Feature feature = cascade.getFeatures()[0];
        assertThat(feature.getRects()).hasSize(2);
        assertThat(feature.getRects()[0].getWeight()).isEqualTo(-1.0f);
        assertThat(feature.getRects()[1].getWeight()).isEqualTo(1.0f);
    }

    @Test
    @DisplayName("raises a clear error when the resource is missing")
    void missingResource() {
        assertThatIOException()
                .isThrownBy(() -> HaarCascadeLoader.loadFromClasspath("cascades/does-not-exist.xml"))
                .withMessageContaining("not found on classpath");
    }

    @Test
    @DisplayName("rejects malformed XML")
    void malformedXml() {
        String xml = "<not a valid cascade";
        assertThatThrownBy(() ->
                HaarCascadeLoader.load(new ByteArrayInputStream(xml.getBytes(StandardCharsets.UTF_8))))
                .isInstanceOf(IOException.class);
    }
}
