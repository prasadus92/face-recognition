package com.facerecognition.infrastructure.persistence;

import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.testing.SyntheticFaces;
import com.facerecognition.domain.model.FaceImage;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectOutputStream;
import java.io.Serializable;
import java.net.URI;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.zip.GZIPOutputStream;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

@DisplayName("ModelCodec")
class ModelCodecTest {

    @Test
    @DisplayName("round-trips a trained extractor, identities and samples")
    void roundTrip() throws Exception {
        EigenfacesExtractor extractor = new EigenfacesExtractor(3);
        List<FaceImage> faces = List.of(
                SyntheticFaces.face(0, 0), SyntheticFaces.face(0, 1),
                SyntheticFaces.face(1, 0), SyntheticFaces.face(1, 1));
        extractor.train(faces, List.of("a", "a", "b", "b"));
        FaceImage probe = SyntheticFaces.face(1, 7);

        TrainedModel model = new TrainedModel(extractor,
                List.of(new TrainedModel.IdentityRecord("a", "Alice", "ext-1", false, Map.of("team", "red"))),
                List.of(new TrainedModel.FaceSample("a", 2, 1, new byte[]{1, (byte) 200})),
                Instant.parse("2026-11-16T00:00:00Z"));

        TrainedModel decoded = ModelCodec.decode(ModelCodec.encode(model));

        assertThat(decoded.extractor().isTrained()).isTrue();
        assertThat(decoded.extractor().extract(probe).getFeatures())
                .containsExactly(extractor.extract(probe).getFeatures());
        assertThat(decoded.identities()).containsExactlyElementsOf(model.identities());
        assertThat(decoded.samples()).singleElement()
                .satisfies(s -> assertThat(s.pixels()).containsExactly(1, (byte) 200));
        assertThat(decoded.createdAt()).isEqualTo(model.createdAt());
    }

    @Test
    @DisplayName("rejects bytes without the FRM header")
    void rejectsMissingMagic() {
        assertThatThrownBy(() -> ModelCodec.decode("not a model".getBytes()))
                .isInstanceOf(InvalidModelException.class)
                .hasMessageContaining("Not a face-recognition model file");
    }

    @Test
    @DisplayName("rejects other format versions with a retrain hint")
    void rejectsOtherVersions() {
        byte[] data = ModelCodec.encode(emptyModel());
        data[3] = (byte) (ModelCodec.FORMAT_VERSION - 1);

        assertThatThrownBy(() -> ModelCodec.decode(data))
                .isInstanceOf(InvalidModelException.class)
                .hasMessageContaining("retrain");
    }

    @Test
    @DisplayName("rejects a truncated file")
    void rejectsTruncated() {
        byte[] data = ModelCodec.encode(emptyModel());

        assertThatThrownBy(() -> ModelCodec.decode(Arrays.copyOf(data, data.length / 2)))
                .isInstanceOf(InvalidModelException.class);
    }

    @Test
    @DisplayName("refuses to instantiate classes outside the allowlist, even when nested")
    void rejectsDisallowedClasses() throws Exception {
        List<Serializable> payload = new ArrayList<>();
        payload.add(URI.create("http://example.invalid").toURL());

        assertThatThrownBy(() -> ModelCodec.decode(rawModelFile(payload)))
                .isInstanceOf(InvalidModelException.class)
                .hasMessageContaining("rejected");
    }

    @Test
    @DisplayName("rejects a well-formed stream that holds something other than a model")
    void rejectsWrongRootType() throws Exception {
        assertThatThrownBy(() -> ModelCodec.decode(rawModelFile("just a string")))
                .isInstanceOf(InvalidModelException.class)
                .hasMessageContaining("does not contain a trained model");
    }

    private static TrainedModel emptyModel() {
        return new TrainedModel(new EigenfacesExtractor(), List.of(), List.of(), Instant.EPOCH);
    }

    private static byte[] rawModelFile(Object root) throws IOException {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        bytes.writeBytes(new byte[]{'F', 'R', 'M', ModelCodec.FORMAT_VERSION});
        try (ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(bytes))) {
            out.writeObject(root);
        }
        return bytes.toByteArray();
    }
}
