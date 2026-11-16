package com.facerecognition.infrastructure.persistence;

import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.infrastructure.extraction.EigenfacesExtractor;
import com.facerecognition.infrastructure.extraction.FisherfacesExtractor;
import com.facerecognition.infrastructure.extraction.LBPHExtractor;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputFilter;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Reads and writes {@link TrainedModel} files.
 *
 * <p>Layout: the 3 ASCII bytes {@code FRM}, 1 format-version byte, then a GZIP stream holding
 * the Java-serialized model. GZIP's CRC-32 catches corruption. Decoding runs behind an
 * {@link ObjectInputFilter} allowlist of the exact classes a model contains, with size limits,
 * so a crafted upload cannot instantiate arbitrary classes or exhaust memory.</p>
 */
public final class ModelCodec {

    private static final Logger LOG = LoggerFactory.getLogger(ModelCodec.class);

    static final byte FORMAT_VERSION = 3;

    private static final byte[] MAGIC = {'F', 'R', 'M'};
    private static final int HEADER_LENGTH = MAGIC.length + 1;

    static final long MAX_STREAM_BYTES = 64L * 1024 * 1024;
    static final long MAX_DEPTH = 32;
    static final long MAX_REFERENCES = 1_000_000;
    static final long MAX_ARRAY_LENGTH = (long) FaceImage.MAX_DIMENSION * FaceImage.MAX_DIMENSION;

    private static final Set<String> ALLOWED_CLASSES = Set.of(
            "java.lang.Boolean", "java.lang.Double", "java.lang.Integer", "java.lang.Long",
            "java.lang.Number", "java.lang.Object", "java.lang.String", "java.lang.Enum", "java.lang.Record",
            "java.time.Instant", "java.time.LocalDateTime", "java.time.Ser",
            "java.util.ArrayList", "java.util.HashMap", "java.util.LinkedHashMap",
            "java.util.Collections$UnmodifiableMap", "java.util.Map$Entry",
            "java.util.CollSer", "java.util.ImmutableCollections$ListN",
            "java.util.ImmutableCollections$List12", "java.util.ImmutableCollections$MapN",
            "java.util.ImmutableCollections$Map1",
            "Jama.Matrix",
            TrainedModel.class.getName(),
            TrainedModel.IdentityRecord.class.getName(),
            TrainedModel.FaceSample.class.getName(),
            FeatureExtractor.ExtractorConfig.class.getName(),
            EigenfacesExtractor.class.getName(),
            FisherfacesExtractor.class.getName(),
            LBPHExtractor.class.getName());

    private ModelCodec() {
    }

    /**
     * Serializes a model into the on-disk format.
     *
     * @param model the model to write
     * @return the encoded bytes
     * @throws IllegalStateException if the model is too large for {@link #decode(byte[])} to read back
     */
    public static byte[] encode(TrainedModel model) {
        return encode(model, MAX_STREAM_BYTES);
    }

    static byte[] encode(TrainedModel model, long maxSerializedBytes) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        bytes.writeBytes(MAGIC);
        bytes.write(FORMAT_VERSION);
        CountingOutputStream counter;
        try (ObjectOutputStream out = new ObjectOutputStream(
                counter = new CountingOutputStream(new GZIPOutputStream(bytes)))) {
            out.writeObject(model);
            out.flush();
            if (counter.count > maxSerializedBytes) {
                throw new IllegalStateException("Model is " + counter.count / (1024 * 1024)
                        + " MB serialized, over the " + maxSerializedBytes / (1024 * 1024) + " MB a model file may hold");
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Could not encode model", e);
        }
        return bytes.toByteArray();
    }

    private static final class CountingOutputStream extends FilterOutputStream {
        private long count;

        CountingOutputStream(OutputStream out) {
            super(out);
        }

        @Override
        public void write(int b) throws IOException {
            out.write(b);
            count++;
        }

        @Override
        public void write(byte[] b, int off, int len) throws IOException {
            out.write(b, off, len);
            count += len;
        }
    }

    /**
     * Parses bytes produced by {@link #encode(TrainedModel)}.
     *
     * @param data the encoded model
     * @return the decoded model
     * @throws InvalidModelException if the bytes are not a supported, intact model file
     */
    public static TrainedModel decode(byte[] data) throws InvalidModelException {
        if (data.length < HEADER_LENGTH
                || !Arrays.equals(data, 0, MAGIC.length, MAGIC, 0, MAGIC.length)) {
            throw new InvalidModelException("Not a face-recognition model file");
        }
        byte version = data[MAGIC.length];
        if (version != FORMAT_VERSION) {
            throw new InvalidModelException("Model format version " + version
                    + " is not supported (expected " + FORMAT_VERSION + "); retrain and save the model again");
        }
        InputStream payload = new ByteArrayInputStream(data, HEADER_LENGTH, data.length - HEADER_LENGTH);
        try (ObjectInputStream in = new ObjectInputStream(new GZIPInputStream(payload))) {
            in.setObjectInputFilter(ModelCodec::check);
            if (in.readObject() instanceof TrainedModel model) {
                return model;
            }
            throw new InvalidModelException("Model file does not contain a trained model");
        } catch (InvalidModelException e) {
            throw e;
        } catch (IOException | ClassNotFoundException | RuntimeException e) {
            throw new InvalidModelException("Model file is corrupt or was rejected", e);
        }
    }

    static ObjectInputFilter.Status check(ObjectInputFilter.FilterInfo info) {
        if (info.depth() > MAX_DEPTH
                || info.references() > MAX_REFERENCES
                || info.streamBytes() > MAX_STREAM_BYTES
                || info.arrayLength() > MAX_ARRAY_LENGTH) {
            return ObjectInputFilter.Status.REJECTED;
        }
        Class<?> type = info.serialClass();
        if (type == null) {
            return ObjectInputFilter.Status.UNDECIDED;
        }
        while (type.isArray()) {
            type = type.getComponentType();
        }
        if (type.isPrimitive() || ALLOWED_CLASSES.contains(type.getName())) {
            return ObjectInputFilter.Status.ALLOWED;
        }
        LOG.warn("Rejected class {} while decoding a model file", type.getName());
        return ObjectInputFilter.Status.REJECTED;
    }
}
