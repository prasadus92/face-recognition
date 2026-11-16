package com.facerecognition.api.cli;

import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.config.FaceRecognitionProperties;
import com.facerecognition.config.FaceRecognitionServiceFactory;
import com.facerecognition.infrastructure.persistence.AtomicFiles;

import picocli.CommandLine.Option;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Options and model-file handling shared by the commands that run the recognition pipeline. */
public final class PipelineOptions {

    @Option(names = "--detect",
            description = "Detect and crop the largest face before recognition. "
                    + "Leave off for images that are already face crops; use the same setting for every command.")
    private boolean detect;

    /**
     * @param props pipeline settings
     * @return a service without a model repository; model files are read and written explicitly
     */
    public FaceRecognitionService newService(FaceRecognitionProperties props) {
        return FaceRecognitionServiceFactory.service(props,
                detect ? FaceRecognitionServiceFactory.detector(props.getDetection()) : null,
                null);
    }

    /**
     * @param model a file written by {@link #save(FaceRecognitionService, Path)}
     * @return a service restored from that file
     * @throws IOException if the file cannot be read or is not a valid model
     */
    public FaceRecognitionService load(Path model) throws IOException {
        FaceRecognitionService service = newService(new FaceRecognitionProperties());
        service.importModel(Files.readAllBytes(model));
        return service;
    }

    /** Writes the model atomically, so an interrupted save keeps the previous file. */
    public static void save(FaceRecognitionService service, Path model) throws IOException {
        AtomicFiles.write(model, service.exportModel());
    }
}
