package com.facerecognition.api.cli;

import picocli.CommandLine.Option;

/** Adds {@code -h/--help} to a command. */
public final class HelpOption {

    @Option(names = {"-h", "--help"}, usageHelp = true, description = "Show this help and exit.")
    private boolean help;
}
