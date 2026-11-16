package com.facerecognition.infrastructure.detection;

import com.facerecognition.infrastructure.detection.RectangleGrouping.Box;
import com.facerecognition.infrastructure.detection.RectangleGrouping.Group;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("RectangleGrouping (cv::groupRectangles)")
class RectangleGroupingTest {

    @Test
    @DisplayName("keeps a cluster only with more than minNeighbours members")
    void needsMoreThanMinNeighbours() {
        assertThat(RectangleGrouping.group(copies(new Box(10, 10, 50, 50), 3), 3)).isEmpty();
        assertThat(RectangleGrouping.group(copies(new Box(10, 10, 50, 50), 4), 3))
                .singleElement().extracting(Group::neighbours).isEqualTo(4);
    }

    @Test
    @DisplayName("averages similar boxes and keeps dissimilar ones apart")
    void averagesSimilarBoxes() {
        List<Box> boxes = new ArrayList<>(List.of(
                new Box(100, 100, 50, 50), new Box(104, 102, 52, 50), new Box(102, 98, 50, 54)));
        boxes.addAll(copies(new Box(300, 300, 40, 40), 3));

        List<Group> groups = RectangleGrouping.group(boxes, 2);

        assertThat(groups).extracting(Group::box)
                .containsExactlyInAnyOrder(new Box(102, 100, 51, 51), new Box(300, 300, 40, 40));
    }

    @Test
    @DisplayName("drops a weak cluster inside a stronger, larger one")
    void suppressesNestedClusters() {
        List<Box> boxes = new ArrayList<>(copies(new Box(0, 0, 200, 200), 8));
        boxes.addAll(copies(new Box(60, 60, 60, 60), 2));

        assertThat(RectangleGrouping.group(boxes, 1)).extracting(Group::box)
                .containsExactly(new Box(0, 0, 200, 200));
    }

    private static List<Box> copies(Box box, int count) {
        List<Box> boxes = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            boxes.add(box);
        }
        return boxes;
    }
}
