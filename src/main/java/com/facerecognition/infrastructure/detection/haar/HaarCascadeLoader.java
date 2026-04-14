package com.facerecognition.infrastructure.detection.haar;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import javax.xml.stream.XMLInputFactory;
import javax.xml.stream.XMLStreamException;
import javax.xml.stream.XMLStreamReader;
import static javax.xml.stream.XMLStreamConstants.CHARACTERS;
import static javax.xml.stream.XMLStreamConstants.END_ELEMENT;
import static javax.xml.stream.XMLStreamConstants.START_ELEMENT;

/**
 * Loads OpenCV's {@code haarcascade_*.xml} files into a {@link HaarCascade}
 * data object.
 *
 * <p>The OpenCV cascade XML format uses an anonymous list element
 * ({@code <_>}) for stages, weak classifiers, features, and rectangles. The
 * payload elements we care about are:</p>
 *
 * <ul>
 *   <li>{@code <width>} / {@code <height>} — the trained window size</li>
 *   <li>{@code <stages>} — an ordered list of stages</li>
 *   <li>{@code <stages>/<_>/<stageThreshold>} — the rejection threshold</li>
 *   <li>{@code <stages>/<_>/<weakClassifiers>/<_>/<internalNodes>} — four whitespace-separated numbers:
 *       {@code <unused> <unused> featureIndex nodeThreshold}</li>
 *   <li>{@code <stages>/<_>/<weakClassifiers>/<_>/<leafValues>} — two whitespace-separated floats</li>
 *   <li>{@code <features>} — the feature pool, indexed by position</li>
 *   <li>{@code <features>/<_>/<rects>/<_>} — text of the form {@code x y w h weight}</li>
 * </ul>
 *
 * <p>This loader is deliberately streaming (StAX) so that the 900 KB default
 * cascade parses into a ~200 KB in-memory representation in well under 100 ms
 * without pulling any XML library beyond the JDK.</p>
 */
public final class HaarCascadeLoader {

    private static final XMLInputFactory FACTORY;
    static {
        FACTORY = XMLInputFactory.newFactory();
        // Harden the parser against malicious cascade files: disable DTDs and
        // external entity resolution entirely. The cascade schema uses neither.
        FACTORY.setProperty(XMLInputFactory.SUPPORT_DTD, Boolean.FALSE);
        FACTORY.setProperty("javax.xml.stream.isSupportingExternalEntities", Boolean.FALSE);
    }

    private HaarCascadeLoader() {
    }

    /** Loads a cascade from a classpath resource. */
    public static HaarCascade loadFromClasspath(String resourcePath) throws IOException {
        ClassLoader cl = Thread.currentThread().getContextClassLoader();
        if (cl == null) {
            cl = HaarCascadeLoader.class.getClassLoader();
        }
        try (InputStream in = cl.getResourceAsStream(resourcePath)) {
            if (in == null) {
                throw new IOException("Haar cascade resource not found on classpath: " + resourcePath);
            }
            return load(in);
        }
    }

    /** Loads a cascade from a local file. */
    public static HaarCascade loadFromFile(Path path) throws IOException {
        try (InputStream in = Files.newInputStream(path)) {
            return load(in);
        }
    }

    /** Loads a cascade from an already-open input stream. The caller owns the stream. */
    public static HaarCascade load(InputStream stream) throws IOException {
        XMLStreamReader reader = null;
        try {
            reader = FACTORY.createXMLStreamReader(stream);

            int windowWidth = -1;
            int windowHeight = -1;
            List<HaarCascade.Stage> stages = new ArrayList<>();
            List<HaarCascade.Feature> features = new ArrayList<>();

            while (reader.hasNext()) {
                int event = reader.next();
                if (event != START_ELEMENT) {
                    continue;
                }
                String name = reader.getLocalName();
                switch (name) {
                    case "width":
                        windowWidth = Integer.parseInt(readText(reader).trim());
                        break;
                    case "height":
                        windowHeight = Integer.parseInt(readText(reader).trim());
                        break;
                    case "stages":
                        parseStages(reader, stages);
                        break;
                    case "features":
                        parseFeatures(reader, features);
                        break;
                    default:
                        break;
                }
            }

            if (windowWidth < 0 || windowHeight < 0) {
                throw new IOException("Cascade XML missing <width>/<height>");
            }
            if (stages.isEmpty()) {
                throw new IOException("Cascade XML missing <stages>");
            }
            if (features.isEmpty()) {
                throw new IOException("Cascade XML missing <features>");
            }
            return new HaarCascade(
                    windowWidth,
                    windowHeight,
                    stages.toArray(new HaarCascade.Stage[0]),
                    features.toArray(new HaarCascade.Feature[0]));
        } catch (XMLStreamException e) {
            throw new IOException("Failed to parse Haar cascade XML: " + e.getMessage(), e);
        } finally {
            if (reader != null) {
                try {
                    reader.close();
                } catch (XMLStreamException ignored) {
                    // nothing to do
                }
            }
        }
    }

    // ---------------------------------------------------------------------
    // Stage + weak-classifier parsing
    // ---------------------------------------------------------------------

    private static void parseStages(XMLStreamReader reader, List<HaarCascade.Stage> out) throws XMLStreamException {
        // We're positioned at <stages>. Consume up to </stages>.
        while (reader.hasNext()) {
            int event = reader.next();
            if (event == END_ELEMENT && "stages".equals(reader.getLocalName())) {
                return;
            }
            if (event == START_ELEMENT && "_".equals(reader.getLocalName())) {
                out.add(parseStage(reader));
            }
        }
    }

    private static HaarCascade.Stage parseStage(XMLStreamReader reader) throws XMLStreamException {
        float threshold = 0f;
        List<HaarCascade.WeakClassifier> classifiers = new ArrayList<>();
        while (reader.hasNext()) {
            int event = reader.next();
            if (event == END_ELEMENT && "_".equals(reader.getLocalName())) {
                break;
            }
            if (event == START_ELEMENT) {
                String name = reader.getLocalName();
                if ("stageThreshold".equals(name)) {
                    threshold = Float.parseFloat(readText(reader).trim());
                } else if ("weakClassifiers".equals(name)) {
                    parseWeakClassifiers(reader, classifiers);
                }
            }
        }
        return new HaarCascade.Stage(threshold, classifiers.toArray(new HaarCascade.WeakClassifier[0]));
    }

    private static void parseWeakClassifiers(XMLStreamReader reader, List<HaarCascade.WeakClassifier> out)
            throws XMLStreamException {
        while (reader.hasNext()) {
            int event = reader.next();
            if (event == END_ELEMENT && "weakClassifiers".equals(reader.getLocalName())) {
                return;
            }
            if (event == START_ELEMENT && "_".equals(reader.getLocalName())) {
                out.add(parseWeakClassifier(reader));
            }
        }
    }

    private static HaarCascade.WeakClassifier parseWeakClassifier(XMLStreamReader reader) throws XMLStreamException {
        int featureIndex = -1;
        float nodeThreshold = 0f;
        float leftLeaf = 0f;
        float rightLeaf = 0f;
        while (reader.hasNext()) {
            int event = reader.next();
            if (event == END_ELEMENT && "_".equals(reader.getLocalName())) {
                break;
            }
            if (event == START_ELEMENT) {
                String name = reader.getLocalName();
                if ("internalNodes".equals(name)) {
                    // "0 -1 FEATURE_INDEX NODE_THRESHOLD"
                    String[] parts = splitOnWhitespace(readText(reader));
                    if (parts.length < 4) {
                        throw new XMLStreamException(
                                "internalNodes expected 4 numbers, got " + parts.length);
                    }
                    featureIndex = Integer.parseInt(parts[2]);
                    nodeThreshold = Float.parseFloat(parts[3]);
                } else if ("leafValues".equals(name)) {
                    // "LEFT_LEAF RIGHT_LEAF"
                    String[] parts = splitOnWhitespace(readText(reader));
                    if (parts.length < 2) {
                        throw new XMLStreamException(
                                "leafValues expected 2 numbers, got " + parts.length);
                    }
                    leftLeaf = Float.parseFloat(parts[0]);
                    rightLeaf = Float.parseFloat(parts[1]);
                }
            }
        }
        return new HaarCascade.WeakClassifier(featureIndex, nodeThreshold, leftLeaf, rightLeaf);
    }

    // ---------------------------------------------------------------------
    // Feature parsing
    // ---------------------------------------------------------------------

    private static void parseFeatures(XMLStreamReader reader, List<HaarCascade.Feature> out)
            throws XMLStreamException {
        while (reader.hasNext()) {
            int event = reader.next();
            if (event == END_ELEMENT && "features".equals(reader.getLocalName())) {
                return;
            }
            if (event == START_ELEMENT && "_".equals(reader.getLocalName())) {
                out.add(parseFeature(reader));
            }
        }
    }

    private static HaarCascade.Feature parseFeature(XMLStreamReader reader) throws XMLStreamException {
        List<HaarCascade.Rect> rects = new ArrayList<>(3);
        while (reader.hasNext()) {
            int event = reader.next();
            if (event == END_ELEMENT && "_".equals(reader.getLocalName())) {
                break;
            }
            if (event == START_ELEMENT && "rects".equals(reader.getLocalName())) {
                parseRects(reader, rects);
            }
        }
        return new HaarCascade.Feature(rects.toArray(new HaarCascade.Rect[0]));
    }

    private static void parseRects(XMLStreamReader reader, List<HaarCascade.Rect> out) throws XMLStreamException {
        while (reader.hasNext()) {
            int event = reader.next();
            if (event == END_ELEMENT && "rects".equals(reader.getLocalName())) {
                return;
            }
            if (event == START_ELEMENT && "_".equals(reader.getLocalName())) {
                String text = readText(reader);
                // Format: "X Y W H WEIGHT"
                String[] parts = splitOnWhitespace(text);
                if (parts.length < 5) {
                    throw new XMLStreamException(
                            "rect expected 5 numbers (x y w h weight), got " + parts.length);
                }
                int x = Integer.parseInt(parts[0]);
                int y = Integer.parseInt(parts[1]);
                int w = Integer.parseInt(parts[2]);
                int h = Integer.parseInt(parts[3]);
                // Weight often has a trailing ".". Parse as double to be tolerant.
                float weight = (float) Double.parseDouble(parts[4]);
                out.add(new HaarCascade.Rect(x, y, w, h, weight));
            }
        }
    }

    // ---------------------------------------------------------------------
    // Text helpers
    // ---------------------------------------------------------------------

    /**
     * Reads the CHARACTERS payload of the current element. Unlike
     * {@link XMLStreamReader#getElementText()}, this is tolerant of
     * whitespace-only segments split across multiple CHARACTERS events, which
     * OpenCV cascade XML frequently produces.
     */
    private static String readText(XMLStreamReader reader) throws XMLStreamException {
        StringBuilder sb = new StringBuilder();
        while (reader.hasNext()) {
            int event = reader.next();
            if (event == CHARACTERS) {
                sb.append(reader.getText());
            } else if (event == END_ELEMENT) {
                return sb.toString();
            } else if (event == START_ELEMENT) {
                // Unexpected nested element — bail.
                throw new XMLStreamException(
                        "unexpected nested element <" + reader.getLocalName() + "> inside text node");
            }
        }
        return sb.toString();
    }

    private static String[] splitOnWhitespace(String s) {
        // The text nodes can contain leading / trailing newlines and runs of
        // whitespace. Use the simple whitespace split.
        return s.trim().split("\\s+");
    }
}
