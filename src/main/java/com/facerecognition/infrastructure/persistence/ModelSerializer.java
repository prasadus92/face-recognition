package com.facerecognition.infrastructure.persistence;

import Jama.Matrix;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import java.util.Objects;
import java.util.zip.CRC32;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * Utility class for serializing and deserializing trained face recognition models.
 *
 * <p>This class provides efficient serialization methods optimized for machine
 * learning model data, including:</p>
 *
 * <ul>
 *   <li>Custom serialization for JAMA Matrix objects</li>
 *   <li>Efficient binary storage format</li>
 *   <li>GZIP compression support</li>
 *   <li>Checksum validation for data integrity</li>
 *   <li>Backward compatibility handling</li>
 * </ul>
 *
 * <h3>File Format:</h3>
 * <p>The binary format consists of:</p>
 * <pre>
 * [Magic bytes: 4 bytes]
 * [Format version: 4 bytes]
 * [Flags: 4 bytes (compression, etc.)]
 * [Header checksum: 8 bytes]
 * [Data length: 8 bytes]
 * [Serialized data: variable]
 * [Data checksum: 8 bytes]
 * </pre>
 *
 * <h3>Usage Example:</h3>
 * <pre>{@code
 * // Serialize with compression
 * byte[] data = ModelSerializer.serialize(model, true);
 *
 * // Write to file
 * ModelSerializer.serializeToFile(model, Paths.get("model.frm"), true);
 *
 * // Load from file
 * TrainedModel loaded = ModelSerializer.deserializeFromFile(Paths.get("model.frm"));
 *
 * // Validate checksum
 * boolean valid = ModelSerializer.validateChecksum(Paths.get("model.frm"));
 * }</pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 * @see TrainedModel
 * @see FileModelRepository
 */
public final class ModelSerializer {

    /**
     * Magic bytes identifying a face recognition model file.
     * ASCII: "FRCM" (Face Recognition Compressed Model)
     */
    public static final byte[] MAGIC_BYTES = new byte[]{0x46, 0x52, 0x43, 0x4D};

    /**
     * Current serialization format version.
     */
    public static final int FORMAT_VERSION = 1;

    /**
     * File extension for model files.
     */
    public static final String MODEL_EXTENSION = ".frm";

    /**
     * File extension for metadata JSON files.
     */
    public static final String METADATA_EXTENSION = ".json";

    // Flag constants
    private static final int FLAG_COMPRESSED = 0x01;
    private static final int FLAG_ENCRYPTED = 0x02;  // Reserved for future use

    // Buffer sizes
    private static final int HEADER_SIZE = 28;  // 4 + 4 + 4 + 8 + 8
    private static final int BUFFER_SIZE = 8192;

    /**
     * Private constructor to prevent instantiation.
     */
    private ModelSerializer() {
        throw new UnsupportedOperationException("Utility class cannot be instantiated");
    }

    /**
     * Serializes a TrainedModel to a byte array.
     *
     * @param model the model to serialize
     * @param compress whether to apply GZIP compression
     * @return the serialized bytes
     * @throws SerializationException if serialization fails
     */
    public static byte[] serialize(TrainedModel model, boolean compress) {
        Objects.requireNonNull(model, "Model cannot be null");

        try {
            // Serialize the model object
            byte[] objectData = serializeObject(model);

            // Compress if requested
            byte[] payload = compress ? compressData(objectData) : objectData;

            // Build the final format with header
            return buildSerializedFormat(payload, compress);

        } catch (IOException e) {
            throw new SerializationException("Failed to serialize model", e);
        }
    }

    /**
     * Deserializes a TrainedModel from a byte array.
     *
     * @param data the serialized data
     * @return the deserialized model
     * @throws SerializationException if deserialization fails
     */
    public static TrainedModel deserialize(byte[] data) {
        Objects.requireNonNull(data, "Data cannot be null");

        try {
            // Verify and parse header
            HeaderInfo header = parseHeader(data);

            // Extract payload
            byte[] payload = Arrays.copyOfRange(data, HEADER_SIZE, HEADER_SIZE + (int) header.dataLength);

            // Verify data checksum
            long actualChecksum = calculateCRC32(payload);
            long expectedChecksum = readLong(data, HEADER_SIZE + (int) header.dataLength);
            if (actualChecksum != expectedChecksum) {
                throw new SerializationException("Data checksum mismatch");
            }

            // Decompress if necessary
            byte[] objectData = header.compressed ? decompressData(payload) : payload;

            // Deserialize the object
            return deserializeObject(objectData);

        } catch (IOException | ClassNotFoundException e) {
            throw new SerializationException("Failed to deserialize model", e);
        }
    }

    /**
     * Serializes a TrainedModel directly to a file.
     *
     * @param model the model to serialize
     * @param file the destination file
     * @param compress whether to apply GZIP compression
     * @throws IOException if writing fails
     */
    public static void serializeToFile(TrainedModel model, java.nio.file.Path file, boolean compress)
            throws IOException {
        byte[] data = serialize(model, compress);
        java.nio.file.Files.write(file, data);
    }

    /**
     * Deserializes a TrainedModel from a file.
     *
     * @param file the source file
     * @return the deserialized model
     * @throws IOException if reading fails
     * @throws SerializationException if deserialization fails
     */
    public static TrainedModel deserializeFromFile(java.nio.file.Path file) throws IOException {
        byte[] data = java.nio.file.Files.readAllBytes(file);
        return deserialize(data);
    }

    /**
     * Validates the checksum of a model file.
     *
     * @param file the file to validate
     * @return true if the checksum is valid
     * @throws IOException if reading fails
     */
    public static boolean validateChecksum(java.nio.file.Path file) throws IOException {
        byte[] data = java.nio.file.Files.readAllBytes(file);
        return validateChecksum(data);
    }

    /**
     * Validates the checksum of serialized data.
     *
     * @param data the serialized data
     * @return true if the checksum is valid
     */
    public static boolean validateChecksum(byte[] data) {
        try {
            HeaderInfo header = parseHeader(data);

            // Verify header checksum
            long expectedHeaderChecksum = readLong(data, 12);
            long actualHeaderChecksum = calculateCRC32(Arrays.copyOfRange(data, 0, 12));
            if (expectedHeaderChecksum != actualHeaderChecksum) {
                return false;
            }

            // Verify data checksum
            byte[] payload = Arrays.copyOfRange(data, HEADER_SIZE, HEADER_SIZE + (int) header.dataLength);
            long expectedDataChecksum = readLong(data, HEADER_SIZE + (int) header.dataLength);
            long actualDataChecksum = calculateCRC32(payload);

            return expectedDataChecksum == actualDataChecksum;

        } catch (Exception e) {
            return false;
        }
    }

    /**
     * Computes the SHA-256 hash of serialized data.
     *
     * @param data the data to hash
     * @return the hash as a hex string
     */
    public static String computeSHA256(byte[] data) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(data);
            return bytesToHex(hash);
        } catch (NoSuchAlgorithmException e) {
            throw new SerializationException("SHA-256 not available", e);
        }
    }

    /**
     * Computes the SHA-256 hash of a file.
     *
     * @param file the file to hash
     * @return the hash as a hex string
     * @throws IOException if reading fails
     */
    public static String computeSHA256(java.nio.file.Path file) throws IOException {
        byte[] data = java.nio.file.Files.readAllBytes(file);
        return computeSHA256(data);
    }

    /**
     * Checks if a file is a valid model file by checking magic bytes.
     *
     * @param file the file to check
     * @return true if the file appears to be a model file
     * @throws IOException if reading fails
     */
    public static boolean isModelFile(java.nio.file.Path file) throws IOException {
        if (!java.nio.file.Files.exists(file) || java.nio.file.Files.size(file) < HEADER_SIZE) {
            return false;
        }

        try (InputStream is = java.nio.file.Files.newInputStream(file)) {
            byte[] magic = new byte[4];
            if (is.read(magic) != 4) {
                return false;
            }
            return Arrays.equals(magic, MAGIC_BYTES);
        }
    }

    /**
     * Gets header information from a model file without loading the entire model.
     *
     * @param file the model file
     * @return header information
     * @throws IOException if reading fails
     * @throws SerializationException if the file is not a valid model file
     */
    public static HeaderInfo getHeaderInfo(java.nio.file.Path file) throws IOException {
        byte[] headerData = new byte[HEADER_SIZE];
        try (InputStream is = java.nio.file.Files.newInputStream(file)) {
            if (is.read(headerData) != HEADER_SIZE) {
                throw new SerializationException("File too small to be a valid model file");
            }
        }
        return parseHeader(headerData);
    }

    // ========================================================================
    // Matrix Serialization Methods
    // ========================================================================

    /**
     * Serializes a JAMA Matrix to a byte array.
     *
     * <p>Format: [rows: 4 bytes][cols: 4 bytes][data: rows*cols*8 bytes]</p>
     *
     * @param matrix the matrix to serialize
     * @return the serialized bytes
     */
    public static byte[] serializeMatrix(Matrix matrix) {
        if (matrix == null) {
            return new byte[]{0, 0, 0, 0, 0, 0, 0, 0}; // Zero dimensions = null marker
        }

        int rows = matrix.getRowDimension();
        int cols = matrix.getColumnDimension();
        double[][] data = matrix.getArray();

        ByteBuffer buffer = ByteBuffer.allocate(8 + rows * cols * 8);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putInt(rows);
        buffer.putInt(cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                buffer.putDouble(data[i][j]);
            }
        }

        return buffer.array();
    }

    /**
     * Deserializes a JAMA Matrix from a byte array.
     *
     * @param data the serialized matrix data
     * @return the deserialized matrix, or null if the data represents a null matrix
     */
    public static Matrix deserializeMatrix(byte[] data) {
        ByteBuffer buffer = ByteBuffer.wrap(data);
        buffer.order(ByteOrder.BIG_ENDIAN);

        int rows = buffer.getInt();
        int cols = buffer.getInt();

        if (rows == 0 && cols == 0) {
            return null;
        }

        double[][] matrixData = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrixData[i][j] = buffer.getDouble();
            }
        }

        return new Matrix(matrixData);
    }

    /**
     * Serializes a 2D double array to bytes.
     *
     * @param array the array to serialize
     * @return the serialized bytes
     */
    public static byte[] serializeDoubleArray2D(double[][] array) {
        if (array == null || array.length == 0) {
            return new byte[]{0, 0, 0, 0, 0, 0, 0, 0};
        }

        int rows = array.length;
        int cols = array[0].length;

        ByteBuffer buffer = ByteBuffer.allocate(8 + rows * cols * 8);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putInt(rows);
        buffer.putInt(cols);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                buffer.putDouble(array[i][j]);
            }
        }

        return buffer.array();
    }

    /**
     * Deserializes a 2D double array from bytes.
     *
     * @param data the serialized data
     * @return the deserialized array
     */
    public static double[][] deserializeDoubleArray2D(byte[] data) {
        ByteBuffer buffer = ByteBuffer.wrap(data);
        buffer.order(ByteOrder.BIG_ENDIAN);

        int rows = buffer.getInt();
        int cols = buffer.getInt();

        if (rows == 0 && cols == 0) {
            return null;
        }

        double[][] array = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                array[i][j] = buffer.getDouble();
            }
        }

        return array;
    }

    /**
     * Serializes a 1D double array to bytes.
     *
     * @param array the array to serialize
     * @return the serialized bytes
     */
    public static byte[] serializeDoubleArray(double[] array) {
        if (array == null || array.length == 0) {
            return new byte[]{0, 0, 0, 0};
        }

        ByteBuffer buffer = ByteBuffer.allocate(4 + array.length * 8);
        buffer.order(ByteOrder.BIG_ENDIAN);
        buffer.putInt(array.length);

        for (double value : array) {
            buffer.putDouble(value);
        }

        return buffer.array();
    }

    /**
     * Deserializes a 1D double array from bytes.
     *
     * @param data the serialized data
     * @return the deserialized array
     */
    public static double[] deserializeDoubleArray(byte[] data) {
        ByteBuffer buffer = ByteBuffer.wrap(data);
        buffer.order(ByteOrder.BIG_ENDIAN);

        int length = buffer.getInt();
        if (length == 0) {
            return null;
        }

        double[] array = new double[length];
        for (int i = 0; i < length; i++) {
            array[i] = buffer.getDouble();
        }

        return array;
    }

    // ========================================================================
    // Compression Methods
    // ========================================================================

    /**
     * Compresses data using GZIP.
     *
     * @param data the data to compress
     * @return the compressed data
     * @throws IOException if compression fails
     */
    public static byte[] compressData(byte[] data) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (GZIPOutputStream gzip = new GZIPOutputStream(baos, BUFFER_SIZE)) {
            gzip.write(data);
        }
        return baos.toByteArray();
    }

    /**
     * Decompresses GZIP-compressed data.
     *
     * @param compressedData the compressed data
     * @return the decompressed data
     * @throws IOException if decompression fails
     */
    public static byte[] decompressData(byte[] compressedData) throws IOException {
        ByteArrayInputStream bais = new ByteArrayInputStream(compressedData);
        ByteArrayOutputStream baos = new ByteArrayOutputStream();

        try (GZIPInputStream gzip = new GZIPInputStream(bais, BUFFER_SIZE)) {
            byte[] buffer = new byte[BUFFER_SIZE];
            int bytesRead;
            while ((bytesRead = gzip.read(buffer)) != -1) {
                baos.write(buffer, 0, bytesRead);
            }
        }

        return baos.toByteArray();
    }

    /**
     * Calculates the compression ratio.
     *
     * @param originalSize the original data size
     * @param compressedSize the compressed data size
     * @return the compression ratio (e.g., 0.7 means 30% reduction)
     */
    public static double compressionRatio(long originalSize, long compressedSize) {
        if (originalSize == 0) return 1.0;
        return (double) compressedSize / originalSize;
    }

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    private static byte[] serializeObject(Object obj) throws IOException {
        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(baos)) {
            oos.writeObject(obj);
        }
        return baos.toByteArray();
    }

    @SuppressWarnings("unchecked")
    private static <T> T deserializeObject(byte[] data) throws IOException, ClassNotFoundException {
        ByteArrayInputStream bais = new ByteArrayInputStream(data);
        try (ObjectInputStream ois = new ObjectInputStream(bais)) {
            return (T) ois.readObject();
        }
    }

    private static byte[] buildSerializedFormat(byte[] payload, boolean compressed) {
        int totalSize = HEADER_SIZE + payload.length + 8; // Header + payload + final checksum
        ByteBuffer buffer = ByteBuffer.allocate(totalSize);
        buffer.order(ByteOrder.BIG_ENDIAN);

        // Magic bytes
        buffer.put(MAGIC_BYTES);

        // Format version
        buffer.putInt(FORMAT_VERSION);

        // Flags
        int flags = compressed ? FLAG_COMPRESSED : 0;
        buffer.putInt(flags);

        // Calculate and write header checksum (of first 12 bytes)
        byte[] partialHeader = Arrays.copyOfRange(buffer.array(), 0, 12);
        buffer.putLong(calculateCRC32(partialHeader));

        // Data length
        buffer.putLong(payload.length);

        // Payload
        buffer.put(payload);

        // Data checksum
        buffer.putLong(calculateCRC32(payload));

        return buffer.array();
    }

    private static HeaderInfo parseHeader(byte[] data) {
        if (data.length < HEADER_SIZE) {
            throw new SerializationException("Data too short to contain valid header");
        }

        ByteBuffer buffer = ByteBuffer.wrap(data);
        buffer.order(ByteOrder.BIG_ENDIAN);

        // Verify magic bytes
        byte[] magic = new byte[4];
        buffer.get(magic);
        if (!Arrays.equals(magic, MAGIC_BYTES)) {
            throw new SerializationException("Invalid magic bytes - not a valid model file");
        }

        // Read version
        int version = buffer.getInt();
        if (version > FORMAT_VERSION) {
            throw new SerializationException(
                    String.format("Unsupported format version: %d (max supported: %d)", version, FORMAT_VERSION));
        }

        // Read flags
        int flags = buffer.getInt();
        boolean compressed = (flags & FLAG_COMPRESSED) != 0;

        // Read header checksum
        long headerChecksum = buffer.getLong();

        // Verify header checksum
        long actualHeaderChecksum = calculateCRC32(Arrays.copyOfRange(data, 0, 12));
        if (headerChecksum != actualHeaderChecksum) {
            throw new SerializationException("Header checksum mismatch");
        }

        // Read data length
        long dataLength = buffer.getLong();

        return new HeaderInfo(version, compressed, dataLength);
    }

    private static long calculateCRC32(byte[] data) {
        CRC32 crc = new CRC32();
        crc.update(data);
        return crc.getValue();
    }

    private static long readLong(byte[] data, int offset) {
        ByteBuffer buffer = ByteBuffer.wrap(data, offset, 8);
        buffer.order(ByteOrder.BIG_ENDIAN);
        return buffer.getLong();
    }

    private static String bytesToHex(byte[] bytes) {
        StringBuilder sb = new StringBuilder();
        for (byte b : bytes) {
            sb.append(String.format("%02x", b));
        }
        return sb.toString();
    }

    // ========================================================================
    // Inner Classes
    // ========================================================================

    /**
     * Information extracted from a model file header.
     */
    public static class HeaderInfo {
        private final int formatVersion;
        private final boolean compressed;
        private final long dataLength;

        /**
         * Creates header information.
         *
         * @param formatVersion the format version
         * @param compressed whether data is compressed
         * @param dataLength the length of the payload data
         */
        public HeaderInfo(int formatVersion, boolean compressed, long dataLength) {
            this.formatVersion = formatVersion;
            this.compressed = compressed;
            this.dataLength = dataLength;
        }

        /**
         * Gets the format version.
         *
         * @return the format version
         */
        public int getFormatVersion() {
            return formatVersion;
        }

        /**
         * Checks if the data is compressed.
         *
         * @return true if compressed
         */
        public boolean isCompressed() {
            return compressed;
        }

        /**
         * Gets the payload data length.
         *
         * @return the data length in bytes
         */
        public long getDataLength() {
            return dataLength;
        }

        @Override
        public String toString() {
            return String.format("HeaderInfo{version=%d, compressed=%s, dataLength=%d}",
                    formatVersion, compressed, dataLength);
        }
    }

    /**
     * Exception thrown when serialization or deserialization fails.
     */
    public static class SerializationException extends RuntimeException {
        private static final long serialVersionUID = 1L;

        /**
         * Creates a new exception.
         *
         * @param message the error message
         */
        public SerializationException(String message) {
            super(message);
        }

        /**
         * Creates a new exception with a cause.
         *
         * @param message the error message
         * @param cause the underlying cause
         */
        public SerializationException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}
