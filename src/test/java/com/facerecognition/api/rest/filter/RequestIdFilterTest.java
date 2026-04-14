package com.facerecognition.api.rest.filter;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.concurrent.atomic.AtomicReference;

import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.slf4j.MDC;
import org.springframework.mock.web.MockFilterChain;
import org.springframework.mock.web.MockHttpServletRequest;
import org.springframework.mock.web.MockHttpServletResponse;

import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

@DisplayName("RequestIdFilter")
class RequestIdFilterTest {

    private final RequestIdFilter filter = new RequestIdFilter();

    @AfterEach
    void cleanMdc() {
        MDC.clear();
    }

    @Test
    @DisplayName("mints a fresh UUID when no request-id header is supplied")
    void mintsFreshId() throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/v1/recognize");
        MockHttpServletResponse response = new MockHttpServletResponse();
        AtomicReference<String> observed = new AtomicReference<>();

        filter.doFilter(request, response, captureMdc(observed));

        String emitted = response.getHeader(RequestIdFilter.HEADER);
        assertThat(emitted).isNotBlank();
        // The MDC value during the downstream chain call should match the header.
        assertThat(observed.get()).isEqualTo(emitted);
        // MDC is cleaned up once the filter returns.
        assertThat(MDC.get(RequestIdFilter.MDC_KEY)).isNull();
    }

    @Test
    @DisplayName("honours an upstream X-Request-ID header")
    void honoursUpstreamRequestId() throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/v1/recognize");
        request.addHeader(RequestIdFilter.HEADER, "upstream-1234");
        MockHttpServletResponse response = new MockHttpServletResponse();
        AtomicReference<String> observed = new AtomicReference<>();

        filter.doFilter(request, response, captureMdc(observed));

        assertThat(response.getHeader(RequestIdFilter.HEADER)).isEqualTo("upstream-1234");
        assertThat(observed.get()).isEqualTo("upstream-1234");
    }

    @Test
    @DisplayName("also accepts an upstream X-Correlation-ID header")
    void honoursCorrelationId() throws Exception {
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/v1/recognize");
        request.addHeader("X-Correlation-ID", "corr-42");
        MockHttpServletResponse response = new MockHttpServletResponse();
        AtomicReference<String> observed = new AtomicReference<>();

        filter.doFilter(request, response, captureMdc(observed));

        assertThat(response.getHeader(RequestIdFilter.HEADER)).isEqualTo("corr-42");
        assertThat(observed.get()).isEqualTo("corr-42");
    }

    @Test
    @DisplayName("cleans up MDC even when the downstream chain throws")
    void cleansUpMdcOnFailure() {
        MockHttpServletRequest request = new MockHttpServletRequest("POST", "/api/v1/recognize");
        MockHttpServletResponse response = new MockHttpServletResponse();

        try {
            filter.doFilter(request, response, (req, res) -> {
                throw new IllegalStateException("boom");
            });
        } catch (Exception ignored) {
            // Swallow — we only care that MDC is clean afterwards.
        }
        assertThat(MDC.get(RequestIdFilter.MDC_KEY)).isNull();
    }

    private MockFilterChain captureMdc(AtomicReference<String> observed) {
        return new MockFilterChain() {
            @Override
            public void doFilter(jakarta.servlet.ServletRequest req, jakarta.servlet.ServletResponse res) {
                observed.set(MDC.get(RequestIdFilter.MDC_KEY));
            }
        };
    }

    @SuppressWarnings("unused")
    private void silenceServletTypes(HttpServletRequest req, HttpServletResponse res) {
        // Keeps the Jakarta servlet imports referenced for IDE tooling.
    }
}
