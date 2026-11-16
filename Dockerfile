# syntax=docker/dockerfile:1

FROM eclipse-temurin:25-jdk-alpine AS builder
WORKDIR /build

COPY mvnw pom.xml ./
COPY .mvn ./.mvn
RUN --mount=type=cache,target=/root/.m2 ./mvnw -B -ntp dependency:go-offline

COPY config ./config
COPY src ./src
RUN --mount=type=cache,target=/root/.m2 ./mvnw -B -ntp -DskipTests package \
 && java -Djarmode=tools -jar target/face-recognition-*-exec.jar extract --layers --launcher --destination /build/layers

FROM eclipse-temurin:25-jre-alpine

LABEL org.opencontainers.image.title="face-recognition" \
      org.opencontainers.image.description="Eigenfaces, Fisherfaces and LBPH face recognition behind a Spring Boot REST API." \
      org.opencontainers.image.source="https://github.com/prasadus92/face-recognition" \
      org.opencontainers.image.licenses="Apache-2.0"

RUN apk add --no-cache tini \
 && addgroup -g 1001 -S app \
 && adduser -u 1001 -S app -G app \
 && mkdir -p /app/data/models /app/logs \
 && chown -R app:app /app
WORKDIR /app

COPY --from=builder --chown=app:app /build/layers/dependencies/ ./
COPY --from=builder --chown=app:app /build/layers/spring-boot-loader/ ./
COPY --from=builder --chown=app:app /build/layers/snapshot-dependencies/ ./
COPY --from=builder --chown=app:app /build/layers/application/ ./

USER app:app
EXPOSE 8080
VOLUME ["/app/data"]

ENV JAVA_TOOL_OPTIONS="-XX:MaxRAMPercentage=75.0 -Djava.awt.headless=true" \
    SPRING_PROFILES_ACTIVE=prod

# Liveness, not overall health: an untrained model reports OUT_OF_SERVICE by design.
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD wget -q -O /dev/null http://localhost:8080/actuator/health/liveness || exit 1

ENTRYPOINT ["/sbin/tini", "--", "java", "org.springframework.boot.loader.launch.JarLauncher"]
