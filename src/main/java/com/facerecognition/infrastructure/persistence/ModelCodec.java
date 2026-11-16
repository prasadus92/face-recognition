package com.facerecognition.infrastructure.persistence;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.ObjectInputFilter;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.UncheckedIOException;
import java.util.Arrays;
import java.util.Set;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Reads and writes {@link TrainedModel} files.
 *
 * <p>Layout: the 3 ASCII bytes {@code FRM}, 1 format-version byte, then a GZIP stream holding
 * the Java-serialized model. GZIP's CRC-32 catches corruption. Decoding runs behind an
 * {@link ObjectInputFilter} allowlist, so a crafted upload cannot instantiate arbitrary classes.</p>
 */
public final class ModelCodec {

    static final byte FORMAT_VERSION = 3;

    private static final byte[] MAGIC = {'F', 'R', 'M'};
    private static final int HEADER_LENGTH = MAGIC.length + 1;
    private static final String PROJECT_PACKAGE = "com.facerecognition.";

    private static final long MAX_STREAM_BYTES = 512L * 1024 * 1024;
    private static final long MAX_DEPTH = 32;
    private static final long MAX_REFERENCES = 2_000_000;
    private static final long MAX_ARRAY_LENGTH = 64L * 1024 * 1024;

    private static final Set<String> ALLOWED_CLASSES = Set.of(
            "java.lang.Boolean", "java.lang.Double", "java.lang.Integer", "java.lang.Long",
            "java.lang.Number", "java.lang.Object", "java.lang.String", "java.lang.Enum", "java.lang.Record",
            "java.time.Instant", "java.time.LocalDateTime", "java.time.Ser",
            "java.util.ArrayList", "java.util.HashMap", "java.util.LinkedHashMap",
            "java.util.Collections$UnmodifiableMap", "java.util.Map$Entry",
            "java.util.CollSer", "java.util.ImmutableCollections$ListN",
            "java.util.ImmutableCollections$List12", "java.util.ImmutableCollections$MapN",
            "java.util.ImmutableCollections$Map1",
            "Jama.Matrix");

    private ModelCodec() {
    }

    /**
     * Serializes a model into the on-disk format.
     *
     * @param model the model to write
     * @return the encoded bytes
     */
    public static byte[] encode(TrainedModel model) {
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        bytes.writeBytes(MAGIC);
        bytes.write(FORMAT_VERSION);
        try (ObjectOutputStream out = new ObjectOutputStream(new GZIPOutputStream(bytes))) {
            out.writeObject(model);
        } catch (IOException e) {
            throw new UncheckedIOException("Could not encode model", e);
        }
        return bytes.toByteArray();
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
            throw new InvalidModelException("Model file is corrupt or was rejected: " + e.getMessage(), e);
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
        return type.isPrimitive() || isAllowed(type.getName())
                ? ObjectInputFilter.Status.ALLOWED
                : ObjectInputFilter.Status.REJECTED;
    }

    private static boolean isAllowed(String className) {
        return className.startsWith(PROJECT_PACKAGE) || ALLOWED_CLASSES.contains(className);
    }
}
