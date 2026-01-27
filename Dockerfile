# Face Recognition API - Docker Image
# ====================================
# Multi-stage build for optimized image size

# Stage 1: Build
FROM maven:3.9-eclipse-temurin-11 AS builder

WORKDIR /app

# Copy pom.xml first for dependency caching
COPY pom.xml .

# Download dependencies (cached if pom.xml unchanged)
RUN mvn dependency:go-offline -B || true

# Copy source code
COPY src ./src

# Build the application
RUN mvn clean package -DskipTests -B

# Stage 2: Runtime
FROM eclipse-temurin:11-jre-alpine

LABEL maintainer="Prasad Subrahmanya <prasadus92@gmail.com>"
LABEL description="Face Recognition API - Production-ready face recognition service"
LABEL version="2.0.0"

# Create non-root user for security
RUN addgroup -g 1001 -S facerecognition && \
    adduser -u 1001 -S facerecognition -G facerecognition

WORKDIR /app

# Create directories for data and logs
RUN mkdir -p /app/data/models /app/logs && \
    chown -R facerecognition:facerecognition /app

# Copy JAR from builder stage
COPY --from=builder /app/target/*.jar app.jar

# Switch to non-root user
USER facerecognition

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD wget --no-verbose --tries=1 --spider http://localhost:8080/actuator/health || exit 1

# JVM options for containers
ENV JAVA_OPTS="-XX:+UseContainerSupport \
    -XX:MaxRAMPercentage=75.0 \
    -XX:+UseG1GC \
    -XX:+HeapDumpOnOutOfMemoryError \
    -Djava.security.egd=file:/dev/./urandom"

# Run the application
ENTRYPOINT ["sh", "-c", "java $JAVA_OPTS -jar app.jar"]
