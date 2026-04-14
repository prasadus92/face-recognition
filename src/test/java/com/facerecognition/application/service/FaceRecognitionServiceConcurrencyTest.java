package com.facerecognition.application.service;

import static org.assertj.core.api.Assertions.assertThat;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.infrastructure.classification.KNNClassifier;
import com.facerecognition.infrastructure.extraction.LBPHExtractor;

/**
 * Stress-tests {@link FaceRecognitionService} thread-safety. Before the
 * ReadWriteLock was introduced, interleaved {@code enroll} / {@code train} /
 * {@code recognize} calls could corrupt the {@code trainingSamples} list and
 * throw {@link java.util.ConcurrentModificationException}. This test locks in
 * the post-refactor guarantee that concurrent access produces consistent
 * final state and never throws.
 */
@DisplayName("FaceRecognitionService concurrency")
class FaceRecognitionServiceConcurrencyTest {

    @Test
    @DisplayName("concurrent enrol + train + recognize never throws and converges to a stable state")
    void concurrentEnrollTrainRecognize() throws Exception {
        // LBPH needs no training pass — ideal for a stress test where we
        // hammer train() over and over. KNN is already ConcurrentHashMap-backed.
        FaceRecognitionService service = FaceRecognitionService.builder()
                .extractor(new LBPHExtractor(4, 4, 1, 8))
                .classifier(new KNNClassifier())
                .config(new FaceRecognitionService.Config()
                        .setTargetWidth(32)
                        .setTargetHeight(32)
                        .setRecognitionThreshold(0.0))
                .build();

        // Pre-seed one identity + train so recognize() has something to match.
        service.enroll(coloured(32, 32, new Color(100, 100, 100)), "seed");
        service.train();

        int enrollers = 4;
        int trainers = 2;
        int recognizers = 4;
        int iterationsPerTask = 25;

        ExecutorService pool = Executors.newFixedThreadPool(enrollers + trainers + recognizers);
        CountDownLatch go = new CountDownLatch(1);
        List<Future<?>> futures = new ArrayList<>();
        AtomicInteger recognizeErrors = new AtomicInteger();

        for (int t = 0; t < enrollers; t++) {
            final int id = t;
            futures.add(pool.submit(() -> {
                go.await();
                for (int i = 0; i < iterationsPerTask; i++) {
                    FaceImage img = coloured(32, 32, new Color((id * 7 + i) % 255, 128, 64));
                    service.enroll(img, "person-" + id + "-" + i);
                }
                return null;
            }));
        }
        for (int t = 0; t < trainers; t++) {
            futures.add(pool.submit(() -> {
                go.await();
                for (int i = 0; i < iterationsPerTask; i++) {
                    try {
                        service.train();
                    } catch (IllegalStateException expected) {
                        // Another thread may have already consumed pending samples.
                    }
                    Thread.sleep(1);
                }
                return null;
            }));
        }
        for (int t = 0; t < recognizers; t++) {
            futures.add(pool.submit(() -> {
                go.await();
                for (int i = 0; i < iterationsPerTask; i++) {
                    try {
                        RecognitionResult result = service.recognize(coloured(32, 32, Color.GRAY));
                        assertThat(result.getStatus()).isNotNull();
                    } catch (IllegalStateException ignored) {
                        // The service may briefly be untrained while a trainer
                        // is mid-flight — that's fine, we just don't count it.
                    } catch (RuntimeException e) {
                        recognizeErrors.incrementAndGet();
                    }
                }
                return null;
            }));
        }

        go.countDown();
        for (Future<?> f : futures) {
            f.get(30, TimeUnit.SECONDS);
        }
        pool.shutdown();
        assertThat(pool.awaitTermination(5, TimeUnit.SECONDS)).isTrue();

        assertThat(recognizeErrors.get())
                .as("recognize() should never fail with an unexpected RuntimeException under contention")
                .isZero();

        // Final training run so the service reaches a quiescent state.
        service.train();
        int totalIdentities = service.getIdentityCount();
        // We pre-seeded 1, and 4 threads enrolled 25 unique names each, so we
        // should end up with 1 + 4*25 = 101 identities.
        assertThat(totalIdentities).isEqualTo(1 + enrollers * iterationsPerTask);

        // Sanity: every enrolled identity made it into the classifier's view.
        for (Identity id : service.getIdentities()) {
            assertThat(id.hasSamples()).isTrue();
        }
    }

    private static FaceImage coloured(int w, int h, Color c) {
        BufferedImage img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = img.createGraphics();
        g.setColor(c);
        g.fillRect(0, 0, w, h);
        g.dispose();
        return FaceImage.fromBufferedImage(img);
    }
}
