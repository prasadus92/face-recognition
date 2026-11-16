package com.facerecognition.api.rest;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("LogValues")
class LogValuesTest {

    @Test
    @DisplayName("replaces line breaks so a value cannot start a forged log line")
    void replacesLineBreaks() {
        assertThat(LogValues.safe("alice\r\nINFO admin logged in")).isEqualTo("alice__INFO admin logged in");
    }

    @Test
    @DisplayName("prints null as null")
    void handlesNull() {
        assertThat(LogValues.safe(null)).isEqualTo("null");
    }
}
