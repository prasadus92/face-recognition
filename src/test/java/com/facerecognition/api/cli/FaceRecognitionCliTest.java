package com.facerecognition.api.cli;

import com.facerecognition.testing.SyntheticFaces;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import picocli.CommandLine;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.stream.Stream;

import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("face-recognition CLI")
class FaceRecognitionCliTest {

    private static final int PEOPLE = 3;
    private static final int IMAGES_EACH = 4;

    @TempDir
    Path dir;

    private Path dataset;
    private Path model;
    private final StringWriter out = new StringWriter();
    private final StringWriter err = new StringWriter();

    @BeforeEach
    void setUp() throws IOException {
        dataset = dir.resolve("dataset");
        model = dir.resolve("models/people.frm");
        for (int person = 0; person < PEOPLE; person++) {
            for (int sample = 0; sample < IMAGES_EACH; sample++) {
                writePng(dataset.resolve("person-" + person).resolve(sample + ".png"), person, sample);
            }
        }
    }

    @Test
    @DisplayName("train writes a model that recognize can use")
    void trainThenRecognize() throws IOException {
        Path probe = writePng(dir.resolve("probe.png"), 1, 99);

        assertThat(run("train", dataset.toString(), "-o", model.toString())).isZero();
        assertThat(out.toString()).contains("Trained Eigenfaces on 12 images of 3 people");

        assertThat(run("recognize", "-m", model.toString(), "-t", "0", probe.toString())).isZero();
        assertThat(out.toString()).contains(probe + "\tperson-1\t");
    }

    @Test
    @DisplayName("train reads PGM files, the format of the AT&T/ORL dataset")
    void trainsOnPgm() throws IOException {
        Path pgmDataset = dir.resolve("pgm");
        for (int person = 0; person < 2; person++) {
            for (int sample = 0; sample < 2; sample++) {
                writePgm(pgmDataset.resolve("s" + person).resolve(sample + ".pgm"), person, sample);
            }
        }

        assertThat(run("train", pgmDataset.toString(), "-o", model.toString(), "-a", "lbph")).isZero();
        assertThat(out.toString()).contains("Trained LBPH on 4 images of 2 people");
    }

    @Test
    @DisplayName("enroll adds a person and retrains the model in place")
    void enrollAddsPerson() throws IOException {
        run("train", dataset.toString(), "-o", model.toString());
        Path first = writePng(dir.resolve("new/0.png"), PEOPLE, 0);
        Path second = writePng(dir.resolve("new/1.png"), PEOPLE, 1);
        Path probe = writePng(dir.resolve("new-probe.png"), PEOPLE, 99);

        assertThat(run("enroll", "-m", model.toString(), "-n", "newcomer", first.toString(), second.toString()))
                .isZero();
        assertThat(out.toString()).contains("The model now knows 4 people");

        run("recognize", "-m", model.toString(), "-t", "0", probe.toString());
        assertThat(out.toString()).contains("\tnewcomer\t");
    }

    @Test
    @DisplayName("recognize walks folders, applies the threshold and can print JSON")
    void recognizeFolderAsJson() throws IOException {
        run("train", dataset.toString(), "-o", model.toString());
        writePng(dir.resolve("probes/a.png"), 0, 50);
        writePng(dir.resolve("probes/nested/b.png"), 2, 50);

        assertThat(run("recognize", "-m", model.toString(), "--json", "-t", "0",
                dir.resolve("probes").toString())).isZero();
        assertThat(out.toString())
                .contains("\"name\" : \"person-0\"")
                .contains("\"name\" : \"person-2\"");

        run("recognize", "-m", model.toString(), "-t", "1", dir.resolve("probes/a.png").toString());
        assertThat(out.toString()).contains("\tunknown\t").doesNotContain("\t0.000");
    }

    @Test
    @DisplayName("recognize reports an unreadable image and still answers for the rest")
    void recognizeKeepsGoingPastBadImages() throws IOException {
        run("train", dataset.toString(), "-o", model.toString());
        writePng(dir.resolve("mixed/good.png"), 2, 60);
        Files.writeString(dir.resolve("mixed/broken.png"), "not an image");

        assertThat(run("recognize", "-m", model.toString(), "-t", "0", dir.resolve("mixed").toString()))
                .isEqualTo(1);
        assertThat(out.toString())
                .contains("broken.png\terror:")
                .contains("good.png\tperson-2\t");
    }

    @Test
    @DisplayName("detect prints the face box and can save an annotated copy")
    void detectAndAnnotate() throws IOException {
        Path portrait = dir.resolve("portrait.png");
        try (var in = getClass().getResourceAsStream("/portrait.png")) {
            Files.copy(in, portrait);
        }
        Path boxed = dir.resolve("boxed.jpg");

        assertThat(run("detect", portrait.toString(), "--annotate", boxed.toString())).isZero();
        assertThat(out.toString()).isEqualTo(portrait + "\t95,88,116,116\t1.00" + System.lineSeparator());
        BufferedImage written = ImageIO.read(boxed.toFile());
        assertThat(written.getWidth()).isEqualTo(300);
        assertThat(written.getHeight()).isEqualTo(330);
        int edge = written.getRGB(95 + 116 / 2, 88);
        assertThat((edge >> 16) & 0xFF).as("red on the box edge").isLessThan(110);
        assertThat(edge & 0xFF).as("blue on the box edge").isGreaterThan(180);

        assertThat(run("detect", "--json", portrait.toString())).isZero();
        assertThat(out.toString()).contains("\"width\" : 116");

        assertThat(run("detect", portrait.toString(), portrait.toString(), "-a", boxed.toString())).isEqualTo(2);
        assertThat(run("detect", portrait.toString(), "-a", dir.resolve("boxed.bmp").toString())).isEqualTo(2);
        assertThat(run("detect", "--min-confidence", "1.5", portrait.toString())).isEqualTo(2);
    }

    @Test
    @DisplayName("detect walks folders and reports an unreadable image without stopping")
    void detectKeepsGoingPastBadImages() throws IOException {
        Path folder = dir.resolve("photos");
        Files.createDirectories(folder);
        try (var in = getClass().getResourceAsStream("/portrait.png")) {
            Files.copy(in, folder.resolve("a.png"));
        }
        Files.writeString(folder.resolve("b.png"), "not an image");

        assertThat(run("detect", folder.toString())).isEqualTo(1);
        assertThat(out.toString())
                .contains("a.png\t95,88,116,116\t1.00")
                .contains("b.png\terror:");
    }

    @Test
    @DisplayName("benchmark writes reports for the chosen algorithms")
    void benchmarkWritesReports() throws IOException {
        Path reports = dir.resolve("reports");

        assertThat(run("benchmark", dataset.toString(), "--type", "ACCURACY", "-a", "eigenfaces,lbph",
                "--train-ratio", "0.5", "-o", reports.toString())).isZero();

        try (Stream<Path> files = Files.walk(reports)) {
            assertThat(files.map(p -> p.getFileName().toString())).contains("benchmark_report.md");
        }
    }

    @Test
    @DisplayName("errors exit 1 with a message; bad usage exits 2")
    void exitCodes() {
        assertThat(run("train", dir.resolve("missing").toString(), "-o", model.toString())).isEqualTo(1);
        assertThat(err.toString()).contains("Error:");

        assertThat(run("train", dataset.toString())).isEqualTo(2);
        assertThat(run("recognize", "-m", model.toString(), "-t", "1.5", "x.png")).isEqualTo(2);
        assertThat(run("benchmark", dataset.toString(), "--format", "imagenet")).isEqualTo(2);
        assertThat(run("recognise", "-m", model.toString())).isEqualTo(2);

        assertThat(run("recognize", "-m", dir.resolve("absent.frm").toString(), "x.png")).isEqualTo(1);
        assertThat(err.toString()).contains("File not found: ");
    }

    @Test
    @DisplayName("--version prints the tool name")
    void version() {
        assertThat(run("--version")).isZero();
        assertThat(out.toString()).startsWith("face-recognition ");
    }

    @Test
    @DisplayName("only CLI commands and help leave the server path")
    void handles() {
        assertThat(FaceRecognitionCli.handles(new String[]{"train", "x"})).isTrue();
        assertThat(FaceRecognitionCli.handles(new String[]{"--help"})).isTrue();
        assertThat(FaceRecognitionCli.handles(new String[]{"help"})).isTrue();
        assertThat(FaceRecognitionCli.handles(new String[]{"--version"})).isTrue();
        assertThat(FaceRecognitionCli.handles(new String[]{"recognise"})).isTrue();
        assertThat(FaceRecognitionCli.handles(new String[]{})).isFalse();
        assertThat(FaceRecognitionCli.handles(new String[]{"--server.port=9090"})).isFalse();
    }

    private int run(String... args) {
        out.getBuffer().setLength(0);
        err.getBuffer().setLength(0);
        CommandLine cli = FaceRecognitionCli.commandLine();
        cli.setOut(new PrintWriter(out, true));
        cli.setErr(new PrintWriter(err, true));
        return cli.execute(args);
    }

    private static Path writePng(Path file, int person, int sample) throws IOException {
        Files.createDirectories(file.getParent());
        ImageIO.write(SyntheticFaces.image(person, sample), "png", file.toFile());
        return file;
    }

    private static void writePgm(Path file, int person, int sample) throws IOException {
        Files.createDirectories(file.getParent());
        byte[] pixels = SyntheticFaces.face(person, sample).toGrayscaleBytes();
        ByteArrayOutputStream bytes = new ByteArrayOutputStream();
        bytes.writeBytes(("P5\n# synthetic\n" + SyntheticFaces.SIZE + " " + SyntheticFaces.SIZE + "\n255\n").getBytes());
        bytes.writeBytes(pixels);
        Files.write(file, bytes.toByteArray());
    }
}
