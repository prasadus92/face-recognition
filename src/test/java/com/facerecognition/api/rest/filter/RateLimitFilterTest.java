package com.facerecognition.api.rest.filter;

import com.facerecognition.config.FaceRecognitionProperties;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

import static org.assertj.core.api.Assertions.assertThat;

@DisplayName("RateLimitFilter")
class RateLimitFilterTest {

    @Test
    @DisplayName("a client cannot get a fresh bucket by sending a new X-Forwarded-For")
    void ignoresClientSuppliedForwardedFor() throws Exception {
        FaceRecognitionProperties properties = new FaceRecognitionProperties();
        properties.getRatelimit().setRequestsPerMinute(1);
        RateLimitFilter filter = new RateLimitFilter(properties);

        assertThat(call(filter, "203.0.113.7", "10.0.0.1")).isEqualTo(200);
        assertThat(call(filter, "203.0.113.7", "10.0.0.2")).isEqualTo(429);
        assertThat(call(filter, "198.51.100.9", "10.0.0.2")).isEqualTo(200);
    }

    private static int call(RateLimitFilter filter, String remoteAddr, String forwardedFor) throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/v1/recognize");
        request.setRemoteAddr(remoteAddr);
        request.addHeader("X-Forwarded-For", forwardedFor);
        MockHttpServletResponse response = new MockHttpServletResponse();
        filter.doFilter(request, response, new MockFilterChain());
        return response.getStatus();
    }
}
