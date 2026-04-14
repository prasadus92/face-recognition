package com.facerecognition.config;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.event.ApplicationReadyEvent;
import org.springframework.context.event.EventListener;
import org.springframework.stereotype.Component;

import com.facerecognition.application.service.FaceRecognitionService;

/**
 * Kicks off {@link FaceRecognitionService#tryLoadSavedModel()} once the Spring
 * context is fully up, so a restart of the server resurrects every enrolled
 * identity without manual intervention.
 *
 * <p>Deliberately a separate component rather than a method on
 * {@code FaceRecognitionApplication}, so that test slices (e.g.
 * {@code @WebMvcTest}) do not transitively require a fully-populated
 * {@link FaceRecognitionProperties} bean.</p>
 */
@Component
@ConditionalOnProperty(name = "facerecognition.model.auto-load", havingValue = "true", matchIfMissing = true)
public class ModelAutoLoadListener {

    private static final Logger log = LoggerFactory.getLogger(ModelAutoLoadListener.class);

    private final FaceRecognitionService service;
    private final FaceRecognitionProperties properties;

    public ModelAutoLoadListener(FaceRecognitionService service,
                                 FaceRecognitionProperties properties) {
        this.service = service;
        this.properties = properties;
    }

    @EventListener(ApplicationReadyEvent.class)
    public void onReady(ApplicationReadyEvent event) {
        boolean restored = service.tryLoadSavedModel();
        if (restored) {
            log.info("Auto-loaded {} identities from '{}'",
                    service.getIdentityCount(), properties.getModel().getSavePath());
        } else {
            log.info("No saved model found at '{}' — starting with an empty pipeline.",
                    properties.getModel().getSavePath());
        }
    }
}
