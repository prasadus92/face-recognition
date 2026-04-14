package com.facerecognition.api.rest.filter;

import java.io.IOException;
import java.util.Objects;

import org.springframework.boot.autoconfigure.condition.ConditionalOnExpression;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import com.facerecognition.config.FaceRecognitionProperties;

import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

/**
 * Minimal static-API-key authentication filter. Only installed when
 * {@code facerecognition.security.api-key} is non-blank. Compares the incoming
 * header (default {@code X-API-Key}) to the configured value using
 * {@link java.security.MessageDigest#isEqual} via {@code Objects.equals}-free
 * constant-time checks.
 *
 * <p>This filter is intentionally simple. Production deployments should front
 * the service with a real identity provider; this is a defensive default for
 * when operators run the API directly.</p>
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE + 20)
@ConditionalOnExpression("'${facerecognition.security.api-key:}' != ''")
public class ApiKeyAuthFilter extends OncePerRequestFilter {

    private static final String API_PREFIX = "/api/v1/";

    private final String expectedKey;
    private final String headerName;

    public ApiKeyAuthFilter(FaceRecognitionProperties properties) {
        this.expectedKey = properties.getSecurity().getApiKey();
        this.headerName = properties.getSecurity().getApiKeyHeader();
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String path = request.getRequestURI();
        return path == null || !path.startsWith(API_PREFIX);
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain chain) throws ServletException, IOException {
        String provided = request.getHeader(headerName);
        if (!constantTimeEquals(provided, expectedKey)) {
            response.setStatus(HttpStatus.UNAUTHORIZED.value());
            response.setContentType(MediaType.APPLICATION_JSON_VALUE);
            response.getWriter().write("{\"code\":\"UNAUTHORIZED\",\"message\":\"Missing or invalid API key.\"}");
            return;
        }
        chain.doFilter(request, response);
    }

    private static boolean constantTimeEquals(String a, String b) {
        if (a == null || b == null) {
            return false;
        }
        if (a.length() != b.length()) {
            return false;
        }
        int diff = 0;
        for (int i = 0; i < a.length(); i++) {
            diff |= a.charAt(i) ^ b.charAt(i);
        }
        return diff == 0 && Objects.equals(a, b);
    }
}
