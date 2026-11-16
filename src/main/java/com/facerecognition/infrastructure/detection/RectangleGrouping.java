package com.facerecognition.infrastructure.detection;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * OpenCV's {@code cv::groupRectangles}: clusters similar detections, keeps clusters with more than
 * {@code minNeighbours} members, averages each cluster, and drops a cluster that sits inside a
 * larger, better-supported one.
 */
final class RectangleGrouping {

    static final double EPS = 0.2;

    /** An axis-aligned box in pixels. */
    record Box(int x, int y, int width, int height) {
    }

    /** A merged box and how many raw detections support it. */
    record Group(Box box, int neighbours) {
    }

    private RectangleGrouping() {
    }

    static List<Group> group(List<Box> boxes, int minNeighbours) {
        int n = boxes.size();
        int[] parent = new int[n];
        for (int i = 0; i < n; i++) {
            parent[i] = i;
        }
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (similar(boxes.get(i), boxes.get(j))) {
                    parent[find(parent, i)] = find(parent, j);
                }
            }
        }

        int[] classOf = new int[n];
        List<long[]> sums = new ArrayList<>();
        int[] rootToClass = new int[n];
        Arrays.fill(rootToClass, -1);
        for (int i = 0; i < n; i++) {
            int root = find(parent, i);
            if (rootToClass[root] < 0) {
                rootToClass[root] = sums.size();
                sums.add(new long[5]);
            }
            classOf[i] = rootToClass[root];
            Box b = boxes.get(i);
            long[] sum = sums.get(classOf[i]);
            sum[0] += b.x();
            sum[1] += b.y();
            sum[2] += b.width();
            sum[3] += b.height();
            sum[4]++;
        }

        List<Group> averaged = new ArrayList<>(sums.size());
        for (long[] sum : sums) {
            double scale = 1.0 / sum[4];
            averaged.add(new Group(new Box(
                    cvRound(sum[0] * scale), cvRound(sum[1] * scale),
                    cvRound(sum[2] * scale), cvRound(sum[3] * scale)), (int) sum[4]));
        }

        List<Group> kept = new ArrayList<>();
        for (int i = 0; i < averaged.size(); i++) {
            Group candidate = averaged.get(i);
            if (candidate.neighbours() <= minNeighbours) {
                continue;
            }
            if (!insideStrongerGroup(i, candidate, averaged, minNeighbours)) {
                kept.add(candidate);
            }
        }
        return kept;
    }

    private static boolean insideStrongerGroup(int index, Group inner, List<Group> groups, int minNeighbours) {
        Box r1 = inner.box();
        for (int j = 0; j < groups.size(); j++) {
            Group outer = groups.get(j);
            if (j == index || outer.neighbours() <= minNeighbours) {
                continue;
            }
            Box r2 = outer.box();
            int dx = cvRound(r2.width() * EPS);
            int dy = cvRound(r2.height() * EPS);
            boolean stronger = outer.neighbours() > Math.max(3, inner.neighbours()) || inner.neighbours() < 3;
            if (stronger
                    && r1.x() >= r2.x() - dx
                    && r1.y() >= r2.y() - dy
                    && r1.x() + r1.width() <= r2.x() + r2.width() + dx
                    && r1.y() + r1.height() <= r2.y() + r2.height() + dy) {
                return true;
            }
        }
        return false;
    }

    private static boolean similar(Box a, Box b) {
        double delta = EPS * (Math.min(a.width(), b.width()) + Math.min(a.height(), b.height())) * 0.5;
        return Math.abs(a.x() - b.x()) <= delta
                && Math.abs(a.y() - b.y()) <= delta
                && Math.abs(a.x() + a.width() - b.x() - b.width()) <= delta
                && Math.abs(a.y() + a.height() - b.y() - b.height()) <= delta;
    }

    /** Round half to even, as OpenCV's {@code cvRound}. */
    private static int cvRound(double value) {
        return (int) Math.rint(value);
    }

    private static int find(int[] parent, int i) {
        while (parent[i] != i) {
            parent[i] = parent[parent[i]];
            i = parent[i];
        }
        return i;
    }
}
