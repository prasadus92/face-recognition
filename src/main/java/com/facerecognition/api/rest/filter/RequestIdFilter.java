package com.facerecognition.api.rest.filter;

import java.io.IOException;
import java.util.UUID;

import org.slf4j.MDC;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

/**
 * Stamps each HTTP request with a correlation ID and exposes it via both the
 * {@code X-Request-ID} response header and SLF4J {@link MDC} (as {@code traceId}).
 * If the caller already supplied an {@code X-Request-ID} / {@code X-Correlation-ID},
 * we honour it; otherwise we mint a fresh UUID.
 *
 * <p>The {@code logback-spring.xml} pattern includes {@code %X{traceId}} so every
 * log line emitted while serving a request carries the ID, which makes correlating
 * upstream requests with internal logs trivial.</p>
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
public class RequestIdFilter extends OncePerRequestFilter {

    /** Canonical header we read from and emit on the response. */
    public static final String HEADER = "X-Request-ID";

    /** MDC key used by the Logback pattern in {@code logback-spring.xml}. */
    public static final String MDC_KEY = "traceId";

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain chain) throws ServletException, IOException {
        String existing = request.getHeader(HEADER);
        if (existing == null || existing.isBlank()) {
            existing = request.getHeader("X-Correlation-ID");
        }
        String traceId = (existing == null || existing.isBlank())
                ? UUID.randomUUID().toString()
                : existing;
        try {
            MDC.put(MDC_KEY, traceId);
            response.setHeader(HEADER, traceId);
            chain.doFilter(request, response);
        } finally {
            MDC.remove(MDC_KEY);
        }
    }
}
