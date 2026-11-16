package com.facerecognition.api.rest.filter;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

import static org.assertj.core.api.Assertions.assertThat;

/** Runs the real server: only Tomcat's proxy handling decides whether X-Forwarded-For is trusted. */
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT, properties = {
    "facerecognition.ratelimit.requests-per-minute=1",
    "facerecognition.model.auto-load=false",
    "facerecognition.model.auto-save=false",
    "server.tomcat.remoteip.internal-proxies=10.9.9.9/32"
})
@DisplayName("Forwarded headers")
class ForwardedHeadersIntegrationTest {

    @LocalServerPort
    private int port;

    @Test
    @DisplayName("X-Forwarded-For from a client that is not a trusted proxy does not change its rate-limit key")
    void untrustedForwardedForIsIgnored() throws Exception {
        assertThat(status("203.0.113.1")).isEqualTo(200);
        assertThat(status("203.0.113.2")).isEqualTo(429);
    }

    private int status(String forwardedFor) throws Exception {
        HttpRequest request = HttpRequest.newBuilder(URI.create("http://localhost:" + port + "/api/v1/model/status"))
                .header("X-Forwarded-For", forwardedFor)
                .GET()
                .build();
        return HttpClient.newHttpClient().send(request, HttpResponse.BodyHandlers.discarding()).statusCode();
    }
}
