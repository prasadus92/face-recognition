package com.facerecognition.api.cli;

import com.facerecognition.api.cli.commands.BenchmarkCommand;
import com.facerecognition.api.cli.commands.EnrollCommand;
import com.facerecognition.api.cli.commands.RecognizeCommand;
import com.facerecognition.api.cli.commands.TrainCommand;

import ch.qos.logback.classic.Level;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Model.CommandSpec;
import picocli.CommandLine.Mixin;
import picocli.CommandLine.Spec;

import java.util.Set;

/** Root of the command-line tool. Running the jar without a command starts the REST API instead. */
@Command(
        name = "face-recognition",
        description = {
            "Train, enroll, recognize and benchmark from the command line.",
            "Run without a command to start the REST API."
        },
        subcommands = {
            TrainCommand.class,
            EnrollCommand.class,
            RecognizeCommand.class,
            BenchmarkCommand.class,
            CommandLine.HelpCommand.class
        })
public final class FaceRecognitionCli implements Runnable {

    private static final Set<String> HELP_FLAGS = Set.of("-h", "--help");

    @Mixin
    private HelpOption help;

    @Spec
    private CommandSpec spec;

    @Override
    public void run() {
        spec.commandLine().usage(spec.commandLine().getOut());
    }

    /**
     * @param args process arguments
     * @return true if the arguments name a CLI command or ask for help, false if they are for the server
     */
    public static boolean handles(String[] args) {
        return args.length > 0
                && (HELP_FLAGS.contains(args[0]) || commandLine().getSubcommands().containsKey(args[0]));
    }

    /**
     * Runs a command and returns its exit code: 0 on success, 1 on error, 2 on bad usage.
     *
     * @param args command and options
     * @return the exit code
     */
    public static int execute(String... args) {
        quietLogging();
        return commandLine().execute(args);
    }

    static CommandLine commandLine() {
        return new CommandLine(new FaceRecognitionCli())
                .setCaseInsensitiveEnumValuesAllowed(true)
                .setExecutionExceptionHandler((ex, cmd, parseResult) -> {
                    cmd.getErr().println(cmd.getColorScheme().errorText("Error: " + ex.getMessage()));
                    return 1;
                });
    }

    private static void quietLogging() {
        if (LoggerFactory.getLogger(Logger.ROOT_LOGGER_NAME) instanceof ch.qos.logback.classic.Logger root) {
            root.setLevel(Level.WARN);
        }
    }
}
