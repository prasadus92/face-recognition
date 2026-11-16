package com.facerecognition.api.cli.commands;

import com.facerecognition.benchmark.BenchmarkRunner.RunnerConfig;
import com.facerecognition.config.FaceRecognitionProperties;
import com.facerecognition.config.FaceRecognitionProperties.ExtractorType;
import com.facerecognition.domain.service.FaceClassifier.DistanceMetric;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import picocli.CommandLine;

import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("benchmark command")
class BenchmarkCommandTest {

    @Test
    @DisplayName("passes the algorithm, components, image size, metric and threshold to the pipeline")
    void mapsOptionsToProperties() {
        BenchmarkCommand command = new BenchmarkCommand();
        new CommandLine(command).setCaseInsensitiveEnumValuesAllowed(true)
                .parseArgs("data", "-m", "chi_square", "-t", "0.25", "-c", "40");
        RunnerConfig config = new RunnerConfig().setNumComponents(40).setRecognitionThreshold(0.25);

        FaceRecognitionProperties props = command.properties(ExtractorType.LBPH, config);

        assertThat(props.getExtraction().getAlgorithm()).isEqualTo(ExtractorType.LBPH);
        assertThat(props.getExtraction().getNumComponents()).isEqualTo(40);
        assertThat(props.getImage().getTargetWidth()).isEqualTo(config.getTargetWidth());
        assertThat(props.getClassification().getDistanceMetric()).isEqualTo(DistanceMetric.CHI_SQUARE);
        assertThat(props.getRecognition().getThreshold()).isEqualTo(0.25);
    }

    @Test
    @DisplayName("defaults to Euclidean distance")
    void defaultsToEuclidean() {
        BenchmarkCommand command = new BenchmarkCommand();
        new CommandLine(command).parseArgs("data");

        assertThat(command.properties(null, new RunnerConfig()).getClassification().getDistanceMetric())
                .isEqualTo(DistanceMetric.EUCLIDEAN);
    }
}
