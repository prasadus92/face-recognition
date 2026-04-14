# syntax=docker/dockerfile:1.6
#
# face-recognition — multi-stage image.
#
# - builder stage: JDK 17 + Maven 3.9, does a clean package and drops tests.
# - runtime stage: JRE 17 (alpine), non-root, container-aware JVM flags,
#   Actuator-backed HEALTHCHECK against /actuator/health.
# Build: docker build -t face-recognition:latest .
# Run:   docker run --rm -p 8080:8080 -v "$PWD/data:/app/data" face-recognition:latest

FROM maven:3.9-eclipse-temurin-17 AS builder
WORKDIR /app

# Pre-fetch dependencies on pom changes so code-only rebuilds are fast.
COPY pom.xml ./
RUN --mount=type=cache,target=/root/.m2 mvn -B -ntp dependency:go-offline || true

COPY src ./src
COPY config ./config
RUN --mount=type=cache,target=/root/.m2 mvn -B -ntp -DskipTests package

# Extract the executable jar (Spring Boot -exec classifier) so the runtime
# stage copies exactly one well-known artifact instead of globbing *.jar.
RUN cp target/face-recognition-*-exec.jar /app/app.jar

# ---------------------------------------------------------------------------

FROM eclipse-temurin:25-jre-alpine

LABEL org.opencontainers.image.title="face-recognition"
LABEL org.opencontainers.image.description="Classical face-recognition library for the JVM — Eigenfaces, Fisherfaces, LBPH — exposed as a Spring Boot REST API."
LABEL org.opencontainers.image.source="https://github.com/prasadus92/face-recognition"
LABEL org.opencontainers.image.licenses="GPL-3.0-only"
LABEL org.opencontainers.image.vendor="Prasad Subrahmanya"

RUN apk add --no-cache wget tini \
 && addgroup -g 1001 -S facerecognition \
 && adduser -u 1001 -S facerecognition -G facerecognition
WORKDIR /app
RUN mkdir -p /app/data/models /app/logs && chown -R facerecognition:facerecognition /app

COPY --from=builder --chown=facerecognition:facerecognition /app/app.jar ./app.jar

USER facerecognition:facerecognition
EXPOSE 8080

ENV JAVA_OPTS="-XX:+UseContainerSupport \
    -XX:MaxRAMPercentage=75.0 \
    -XX:+UseG1GC \
    -XX:+HeapDumpOnOutOfMemoryError \
    -Djava.security.egd=file:/dev/./urandom \
    -Djava.awt.headless=true"
ENV SPRING_PROFILES_ACTIVE=prod

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/actuator/health || exit 1

ENTRYPOINT ["/sbin/tini", "--"]
CMD ["sh", "-c", "exec java $JAVA_OPTS -jar /app/app.jar"]
