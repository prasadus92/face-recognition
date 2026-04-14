package com.facerecognition.api.rest.metrics;

import java.util.concurrent.TimeUnit;

import org.springframework.stereotype.Component;

import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;

/**
 * Centralised Micrometer meters for the face-recognition pipeline.
 *
 * <p>All meters are registered eagerly against the configured
 * {@link MeterRegistry} and exposed under the {@code facerecognition.*}
 * namespace. The Prometheus endpoint at {@code /actuator/prometheus} will
 * pick them up automatically because
 * {@code micrometer-registry-prometheus} is on the classpath.</p>
 */
@Component
public class RecognitionMetrics {

    private final Timer recognizeTotal;
    private final Counter enrollments;
    private final Counter recognitionsSuccessful;
    private final Counter recognitionsFailed;
    private final Counter trainingRuns;
    private final Counter errors;

    public RecognitionMetrics(MeterRegistry registry) {
        this.recognizeTotal = Timer.builder("facerecognition.recognize.total")
                .description("Total wall-clock time of the /recognize endpoint")
                .publishPercentiles(0.5, 0.9, 0.99)
                .register(registry);
        this.enrollments = Counter.builder("facerecognition.enrollments")
                .description("Number of successful enrolments")
                .register(registry);
        this.recognitionsSuccessful = Counter.builder("facerecognition.recognitions")
                .description("Number of recognitions that returned a confident match")
                .tag("outcome", "matched")
                .register(registry);
        this.recognitionsFailed = Counter.builder("facerecognition.recognitions")
                .description("Number of recognitions that did not produce a confident match")
                .tag("outcome", "unmatched")
                .register(registry);
        this.trainingRuns = Counter.builder("facerecognition.training.runs")
                .description("Number of successful training runs")
                .register(registry);
        this.errors = Counter.builder("facerecognition.errors")
                .description("Total number of errors raised by the pipeline")
                .register(registry);
    }

    public void recordRecognize(long nanos, boolean matched) {
        recognizeTotal.record(nanos, TimeUnit.NANOSECONDS);
        if (matched) {
            recognitionsSuccessful.increment();
        } else {
            recognitionsFailed.increment();
        }
    }

    public void recordEnrollment() {
        enrollments.increment();
    }

    public void recordTraining() {
        trainingRuns.increment();
    }

    public void recordError() {
        errors.increment();
    }
}
