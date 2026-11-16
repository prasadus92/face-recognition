package com.facerecognition.api.rest;

import com.facerecognition.api.rest.metrics.RecognitionMetrics;
import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.config.FaceRecognitionProperties;
import com.facerecognition.infrastructure.classification.KNNClassifier;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.infrastructure.persistence.InvalidModelException;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.webmvc.test.autoconfigure.WebMvcTest;
import org.springframework.mock.web.MockMultipartFile;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import java.util.List;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.multipart;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.content;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(controllers = TrainingController.class)
@EnableConfigurationProperties(FaceRecognitionProperties.class)
@DisplayName("TrainingController (MockMvc)")
class TrainingControllerWebMvcTest {

    @Autowired
    private MockMvc mvc;

    @MockitoBean
    private FaceRecognitionService service;

    @MockitoBean
    private RecognitionMetrics metrics;

    @BeforeEach
    void setUp() {
        when(service.getExtractor()).thenReturn(new EigenfacesExtractor());
        when(service.getClassifier()).thenReturn(new KNNClassifier());
        when(service.getConfig()).thenReturn(new FaceRecognitionService.Config());
        when(service.getIdentities()).thenReturn(List.of());
    }

    @Test
    @DisplayName("POST /model/export streams the encoded model")
    void exportsModel() throws Exception {
        when(service.isTrained()).thenReturn(true);
        when(service.exportModel()).thenReturn(new byte[]{'F', 'R', 'M', 3});

        mvc.perform(post("/api/v1/model/export"))
                .andExpect(status().isOk())
                .andExpect(content().bytes(new byte[]{'F', 'R', 'M', 3}));
    }

    @Test
    @DisplayName("POST /model/import hands the upload to the service")
    void importsModel() throws Exception {
        byte[] model = {'F', 'R', 'M', 3, 0};

        mvc.perform(multipart("/api/v1/model/import").file(new MockMultipartFile("file", model)))
                .andExpect(status().isOk());

        verify(service).importModel(model);
    }

    @Test
    @DisplayName("POST /model/import answers 400 for a file the codec rejects")
    void rejectsInvalidModel() throws Exception {
        doThrow(new InvalidModelException("Not a face-recognition model file"))
                .when(service).importModel(any());

        mvc.perform(multipart("/api/v1/model/import")
                        .file(new MockMultipartFile("file", "x.frm", null, "garbage".getBytes())))
                .andExpect(status().isBadRequest())
                .andExpect(jsonPath("$.code").value("IMPORT_FAILED"))
                .andExpect(jsonPath("$.message").value("Model import failed: Not a face-recognition model file"));
    }
}
