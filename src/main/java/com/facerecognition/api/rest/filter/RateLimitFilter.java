package com.facerecognition.api.rest.filter;

import java.io.IOException;
import java.time.Duration;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;

import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.filter.OncePerRequestFilter;

import com.facerecognition.config.FaceRecognitionProperties;

import io.github.bucket4j.Bandwidth;
import io.github.bucket4j.Bucket;
import jakarta.servlet.FilterChain;
import jakarta.servlet.ServletException;
import jakarta.servlet.http.HttpServletRequest;
import jakarta.servlet.http.HttpServletResponse;

/**
 * Per-client-IP token-bucket rate limiter built on Bucket4j.
 *
 * <p>Buckets are keyed by {@link HttpServletRequest#getRemoteAddr()}. With
 * {@code server.forward-headers-strategy: native}, Tomcat sets it from {@code X-Forwarded-For}
 * only when the request comes from an internal proxy address, so clients cannot pick their key.</p>
 *
 * <p>Enabled when {@code facerecognition.ratelimit.enabled=true} (the default).
 * Only the {@code /api/v1/**} surface is rate-limited — actuator, swagger and
 * static resources are deliberately excluded so health checks and docs stay
 * responsive under load.</p>
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE + 10)
@ConditionalOnProperty(name = "facerecognition.ratelimit.enabled", havingValue = "true", matchIfMissing = true)
public class RateLimitFilter extends OncePerRequestFilter {

    private static final String RATE_LIMIT_PATH_PREFIX = "/api/v1/";

    private final ConcurrentMap<String, Bucket> buckets = new ConcurrentHashMap<>();
    private final int requestsPerMinute;

    public RateLimitFilter(FaceRecognitionProperties properties) {
        this.requestsPerMinute = Math.max(1, properties.getRatelimit().getRequestsPerMinute());
    }

    @Override
    protected boolean shouldNotFilter(HttpServletRequest request) {
        String path = request.getRequestURI();
        return path == null || !path.startsWith(RATE_LIMIT_PATH_PREFIX);
    }

    @Override
    protected void doFilterInternal(HttpServletRequest request,
                                    HttpServletResponse response,
                                    FilterChain chain) throws ServletException, IOException {
        Bucket bucket = buckets.computeIfAbsent(request.getRemoteAddr(), k -> newBucket());
        if (bucket.tryConsume(1)) {
            response.setHeader("X-RateLimit-Remaining", Long.toString(bucket.getAvailableTokens()));
            chain.doFilter(request, response);
            return;
        }
        response.setStatus(HttpStatus.TOO_MANY_REQUESTS.value());
        response.setContentType(MediaType.APPLICATION_JSON_VALUE);
        response.setHeader("Retry-After", "60");
        response.getWriter().write("{\"code\":\"RATE_LIMIT_EXCEEDED\","
                + "\"message\":\"Too many requests; slow down.\"}");
    }

    private Bucket newBucket() {
        return Bucket.builder()
                .addLimit(Bandwidth.builder()
                        .capacity(requestsPerMinute)
                        .refillGreedy(requestsPerMinute, Duration.ofMinutes(1))
                        .build())
                .build();
    }
}
