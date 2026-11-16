package com.facerecognition.api.rest;

/** Makes client-supplied values safe to log on one line. */
final class LogValues {

    private LogValues() {
    }

    /**
     * @param value a value that came from the request
     * @return the value with CR and LF replaced, so it cannot forge extra log lines
     */
    static String safe(Object value) {
        return String.valueOf(value).replace('\r', '_').replace('\n', '_');
    }
}
