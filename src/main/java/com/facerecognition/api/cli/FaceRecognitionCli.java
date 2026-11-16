package com.facerecognition.api.cli;

import com.facerecognition.api.cli.commands.BenchmarkCommand;
import com.facerecognition.api.cli.commands.DetectCommand;
import com.facerecognition.api.cli.commands.EnrollCommand;
import com.facerecognition.api.cli.commands.RecognizeCommand;
import com.facerecognition.api.cli.commands.TrainCommand;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.LoggerContext;
import ch.qos.logback.classic.encoder.PatternLayoutEncoder;
import ch.qos.logback.classic.spi.ILoggingEvent;
import ch.qos.logback.core.ConsoleAppender;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.IVersionProvider;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Spec;

import java.nio.file.NoSuchFileException;
import java.util.Objects;
import java.util.Set;

/** Root of the command-line tool. Running the jar without a command starts the REST API instead. */
@Command(
        name = "face-recognition",
        mixinStandardHelpOptions = true,
        versionProvider = FaceRecognitionCli.ManifestVersion.class,
        description = {
            "Train, enroll, recognize, detect and benchmark from the command line.",
            "Run without a command to start the REST API."
        },
        subcommands = {
            TrainCommand.class,
            EnrollCommand.class,
            RecognizeCommand.class,
            DetectCommand.class,
            BenchmarkCommand.class,
            CommandLine.HelpCommand.class
        })
public final class FaceRecognitionCli implements Runnable {

    private static final Set<String> ROOT_FLAGS = Set.of("-h", "--help", "-V", "--version");

    @Spec
    private CommandSpec spec;

    @Override
    public void run() {
        spec.commandLine().usage(spec.commandLine().getOut());
    }

    /**
     * Spring Boot takes only {@code --name=value} arguments, so any other first argument is
     * meant for the CLI, including a mistyped command.
     *
     * @param args process arguments
     * @return true if the CLI should handle these arguments
     */
    public static boolean handles(String[] args) {
        return args.length > 0 && (!args[0].startsWith("-") || ROOT_FLAGS.contains(args[0]));
    }

    /**
     * Runs a command and returns its exit code: 0 on success, 1 on error, 2 on bad usage.
     *
     * @param args command and options
     * @return the exit code
     */
    public static int execute(String... args) {
        logWarningsToStderr();
        return commandLine().execute(args);
    }

    static CommandLine commandLine() {
        return new CommandLine(new FaceRecognitionCli())
                .setCaseInsensitiveEnumValuesAllowed(true)
                .setExecutionExceptionHandler((ex, cmd, parseResult) -> {
                    String message = ex instanceof NoSuchFileException
                            ? "File not found: " + ex.getMessage()
                            : ex.getMessage();
                    cmd.getErr().println(cmd.getColorScheme().errorText("Error: " + message));
                    return 1;
                });
    }

    /** Keeps stdout for command output, so {@code --json} stays parseable. */
    private static void logWarningsToStderr() {
        if (!(LoggerFactory.getILoggerFactory() instanceof LoggerContext context)) {
            return;
        }
        context.reset();
        PatternLayoutEncoder encoder = new PatternLayoutEncoder();
        encoder.setContext(context);
        encoder.setPattern("%level %logger{0}: %msg%n");
        encoder.start();
        ConsoleAppender<ILoggingEvent> stderr = new ConsoleAppender<>();
        stderr.setContext(context);
        stderr.setTarget("System.err");
        stderr.setEncoder(encoder);
        stderr.start();
        ch.qos.logback.classic.Logger root = context.getLogger(Logger.ROOT_LOGGER_NAME);
        root.setLevel(Level.WARN);
        root.addAppender(stderr);
    }

    /** Reads the version Maven writes into the jar manifest. */
    static final class ManifestVersion implements IVersionProvider {
        @Override
        public String[] getVersion() {
            String version = FaceRecognitionCli.class.getPackage().getImplementationVersion();
            return new String[]{"face-recognition " + Objects.requireNonNullElse(version, "(development build)")};
        }
    }
}
