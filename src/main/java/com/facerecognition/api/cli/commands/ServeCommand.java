package com.facerecognition.api.cli.commands;

import com.facerecognition.api.cli.FaceRecognitionCLI;
import com.facerecognition.application.service.FaceRecognitionService;
import com.facerecognition.domain.model.FaceImage;
import com.facerecognition.domain.model.Identity;
import com.facerecognition.domain.model.RecognitionResult;
import com.facerecognition.domain.service.FeatureExtractor;
import com.facerecognition.infrastructure.classification.KNNClassifier;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.fasterxml.jackson.datatype.jsr310.JavaTimeModule;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;

import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.ParentCommand;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.*;
import java.net.InetSocketAddress;
import java.time.LocalDateTime;
import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

/**
 * CLI command for starting a REST API server for face recognition.
 *
 * <p>This command launches an HTTP server that provides RESTful endpoints
 * for face recognition operations. The server supports JSON responses
 * and multipart form data for image uploads.</p>
 *
 * <h3>Available Endpoints:</h3>
 * <ul>
 *   <li><b>GET /health</b> - Health check endpoint</li>
 *   <li><b>GET /info</b> - Model and server information</li>
 *   <li><b>POST /recognize</b> - Recognize faces in uploaded images</li>
 *   <li><b>POST /enroll</b> - Enroll new faces</li>
 *   <li><b>GET /identities</b> - List enrolled identities</li>
 *   <li><b>DELETE /identities/{id}</b> - Remove an identity</li>
 * </ul>
 *
 * <h3>Examples:</h3>
 * <pre>
 * # Start server on default port (8080)
 * face-recognition serve --model model.dat
 *
 * # Start on custom port with verbose logging
 * face-recognition serve --model model.dat --port 9000 -v
 *
 * # Start with CORS enabled for development
 * face-recognition serve --model model.dat --cors
 *
 * # API usage with curl:
 * curl -X POST -F "image=@photo.jpg" http://localhost:8080/recognize
 * curl http://localhost:8080/identities
 * </pre>
 *
 * @author Prasad Subrahmanya
 * @version 2.0
 * @since 2.0
 */
@Command(
    name = "serve",
    aliases = {"server", "api"},
    description = "Start a REST API server for face recognition.",
    mixinStandardHelpOptions = true,
    sortOptions = false,
    footer = {
        "",
        "API Endpoints:",
        "  GET  /health       Health check",
        "  GET  /info         Server and model information",
        "  POST /recognize    Recognize faces (multipart image)",
        "  POST /enroll       Enroll new face (multipart + identity)",
        "  GET  /identities   List all enrolled identities",
        "",
        "Examples:",
        "  @|bold face-recognition serve --model model.dat --port 8080|@",
        "  @|bold curl -F \"image=@photo.jpg\" http://localhost:8080/recognize|@"
    }
)
public class ServeCommand implements Callable<Integer> {

    @ParentCommand
    private FaceRecognitionCLI parent;

    @Option(names = {"-m", "--model"}, paramLabel = "FILE", required = true,
            description = "Path to the trained model file")
    private File modelFile;

    @Option(names = {"-p", "--port"}, paramLabel = "PORT",
            description = "Server port (default: ${DEFAULT-VALUE})",
            defaultValue = "8080")
    private int port;

    @Option(names = {"--host"}, paramLabel = "HOST",
            description = "Server host/interface (default: ${DEFAULT-VALUE})",
            defaultValue = "0.0.0.0")
    private String host;

    @Option(names = {"-t", "--threshold"}, paramLabel = "VALUE",
            description = "Recognition threshold (default: ${DEFAULT-VALUE})",
            defaultValue = "0.6")
    private double threshold;

    @Option(names = {"--threads"}, paramLabel = "N",
            description = "Number of worker threads (default: ${DEFAULT-VALUE})",
            defaultValue = "4")
    private int threads;

    @Option(names = {"--cors"},
            description = "Enable CORS headers for cross-origin requests")
    private boolean enableCors;

    @Option(names = {"--read-only"},
            description = "Disable enrollment endpoints (read-only mode)")
    private boolean readOnly;

    @Option(names = {"-v", "--verbose"},
            description = "Log all requests")
    private boolean verbose;

    @Option(names = {"--api-key"}, paramLabel = "KEY",
            description = "Require API key for authentication")
    private String apiKey;

    @Option(names = {"--max-upload-size"}, paramLabel = "BYTES",
            description = "Maximum upload size in bytes (default: ${DEFAULT-VALUE})",
            defaultValue = "10485760")
    private long maxUploadSize;

    // Server state
    private FaceRecognitionService service;
    private ObjectMapper jsonMapper;
    private AtomicLong requestCounter;
    private LocalDateTime startTime;

    @Override
    public Integer call() throws Exception {
        // Validate model file
        if (!modelFile.exists()) {
            System.err.println("Error: Model file not found: " + modelFile);
            return FaceRecognitionCLI.EXIT_FILE_NOT_FOUND;
        }

        // Load model
        service = loadModel(modelFile);
        if (service == null) {
            return FaceRecognitionCLI.EXIT_ERROR;
        }

        if (!service.isTrained()) {
            System.err.println("Error: Model is not trained");
            return FaceRecognitionCLI.EXIT_MODEL_NOT_TRAINED;
        }

        // Initialize
        jsonMapper = new ObjectMapper();
        jsonMapper.registerModule(new JavaTimeModule());
        jsonMapper.enable(SerializationFeature.INDENT_OUTPUT);
        requestCounter = new AtomicLong(0);
        startTime = LocalDateTime.now();

        // Create HTTP server
        HttpServer server;
        try {
            server = HttpServer.create(new InetSocketAddress(host, port), 0);
        } catch (IOException e) {
            System.err.println("Error starting server: " + e.getMessage());
            return FaceRecognitionCLI.EXIT_ERROR;
        }

        // Configure endpoints
        server.createContext("/health", new HealthHandler());
        server.createContext("/info", new InfoHandler());
        server.createContext("/recognize", new RecognizeHandler());
        server.createContext("/identities", new IdentitiesHandler());

        if (!readOnly) {
            server.createContext("/enroll", new EnrollHandler());
        }

        // Set up thread pool
        server.setExecutor(Executors.newFixedThreadPool(threads));

        // Start server
        server.start();

        System.out.println();
        System.out.println("=".repeat(50));
        System.out.println("Face Recognition API Server");
        System.out.println("=".repeat(50));
        System.out.println("URL: http://" + (host.equals("0.0.0.0") ? "localhost" : host) + ":" + port);
        System.out.println("Model: " + modelFile.getName());
        System.out.println("Identities: " + service.getIdentityCount());
        System.out.println("Threshold: " + threshold);
        System.out.println("Threads: " + threads);
        System.out.println("Read-only: " + readOnly);
        System.out.println("CORS: " + enableCors);
        if (apiKey != null) {
            System.out.println("Authentication: API Key required");
        }
        System.out.println();
        System.out.println("Endpoints:");
        System.out.println("  GET  /health     - Health check");
        System.out.println("  GET  /info       - Server information");
        System.out.println("  POST /recognize  - Recognize faces");
        System.out.println("  GET  /identities - List identities");
        if (!readOnly) {
            System.out.println("  POST /enroll     - Enroll new face");
        }
        System.out.println();
        System.out.println("Press Ctrl+C to stop the server...");

        // Keep running until interrupted
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            System.out.println("\nShutting down server...");
            server.stop(1);
            System.out.println("Server stopped.");
        }));

        // Block forever
        Thread.currentThread().join();

        return FaceRecognitionCLI.EXIT_SUCCESS;
    }

    private FaceRecognitionService loadModel(File modelFile) {
        try {
            System.out.println("Loading model from: " + modelFile);

            try (ObjectInputStream ois = new ObjectInputStream(
                    new BufferedInputStream(new FileInputStream(modelFile)))) {
                Object obj = ois.readObject();

                if (obj instanceof FaceRecognitionService) {
                    FaceRecognitionService svc = (FaceRecognitionService) obj;
                    System.out.println("Model loaded: " + svc.getIdentityCount() + " identities");
                    return svc;
                } else if (obj instanceof FeatureExtractor) {
                    FeatureExtractor extractor = (FeatureExtractor) obj;
                    return FaceRecognitionService.builder()
                        .extractor(extractor)
                        .classifier(new KNNClassifier())
                        .build();
                } else {
                    System.err.println("Error: Unknown model type");
                    return null;
                }
            }
        } catch (Exception e) {
            System.err.println("Error loading model: " + e.getMessage());
            return null;
        }
    }

    private void logRequest(HttpExchange exchange, int statusCode, long durationMs) {
        if (verbose) {
            System.out.printf("[%s] %s %s -> %d (%dms)%n",
                LocalDateTime.now().toString().substring(11, 19),
                exchange.getRequestMethod(),
                exchange.getRequestURI().getPath(),
                statusCode,
                durationMs);
        }
    }

    private boolean checkAuth(HttpExchange exchange) throws IOException {
        if (apiKey == null) return true;

        String authHeader = exchange.getRequestHeaders().getFirst("X-API-Key");
        if (authHeader == null) {
            authHeader = exchange.getRequestHeaders().getFirst("Authorization");
            if (authHeader != null && authHeader.startsWith("Bearer ")) {
                authHeader = authHeader.substring(7);
            }
        }

        if (!apiKey.equals(authHeader)) {
            sendError(exchange, 401, "Unauthorized: Invalid or missing API key");
            return false;
        }
        return true;
    }

    private void addCorsHeaders(HttpExchange exchange) {
        if (enableCors) {
            exchange.getResponseHeaders().add("Access-Control-Allow-Origin", "*");
            exchange.getResponseHeaders().add("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS");
            exchange.getResponseHeaders().add("Access-Control-Allow-Headers", "Content-Type, X-API-Key, Authorization");
        }
    }

    private void sendJson(HttpExchange exchange, int statusCode, Object data) throws IOException {
        addCorsHeaders(exchange);
        exchange.getResponseHeaders().add("Content-Type", "application/json");

        byte[] response = jsonMapper.writeValueAsBytes(data);
        exchange.sendResponseHeaders(statusCode, response.length);

        try (OutputStream os = exchange.getResponseBody()) {
            os.write(response);
        }
    }

    private void sendError(HttpExchange exchange, int statusCode, String message) throws IOException {
        Map<String, Object> error = new LinkedHashMap<>();
        error.put("error", true);
        error.put("status", statusCode);
        error.put("message", message);
        sendJson(exchange, statusCode, error);
    }

    // Handler implementations

    private class HealthHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            long start = System.currentTimeMillis();

            if (exchange.getRequestMethod().equals("OPTIONS")) {
                addCorsHeaders(exchange);
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("status", "healthy");
            response.put("timestamp", LocalDateTime.now().toString());

            sendJson(exchange, 200, response);
            logRequest(exchange, 200, System.currentTimeMillis() - start);
        }
    }

    private class InfoHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            long start = System.currentTimeMillis();

            if (!checkAuth(exchange)) return;

            if (exchange.getRequestMethod().equals("OPTIONS")) {
                addCorsHeaders(exchange);
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("name", "Face Recognition API");
            response.put("version", "2.0.0");
            response.put("startTime", startTime.toString());
            response.put("requests", requestCounter.get());
            response.put("identities", service.getIdentityCount());
            response.put("algorithm", service.getExtractor().getAlgorithmName());
            response.put("threshold", threshold);
            response.put("readOnly", readOnly);

            Map<String, Object> model = new LinkedHashMap<>();
            model.put("file", modelFile.getName());
            model.put("trained", service.isTrained());
            model.put("featureDimension", service.getExtractor().getFeatureDimension());
            response.put("model", model);

            sendJson(exchange, 200, response);
            logRequest(exchange, 200, System.currentTimeMillis() - start);
        }
    }

    private class RecognizeHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            long start = System.currentTimeMillis();
            requestCounter.incrementAndGet();

            if (!checkAuth(exchange)) return;

            if (exchange.getRequestMethod().equals("OPTIONS")) {
                addCorsHeaders(exchange);
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            if (!exchange.getRequestMethod().equals("POST")) {
                sendError(exchange, 405, "Method not allowed. Use POST.");
                logRequest(exchange, 405, System.currentTimeMillis() - start);
                return;
            }

            try {
                // Read image from request body
                BufferedImage image = ImageIO.read(exchange.getRequestBody());
                if (image == null) {
                    sendError(exchange, 400, "Invalid or missing image data");
                    logRequest(exchange, 400, System.currentTimeMillis() - start);
                    return;
                }

                FaceImage faceImage = FaceImage.fromBufferedImage(image);
                RecognitionResult result = service.recognize(faceImage);

                Map<String, Object> response = new LinkedHashMap<>();
                response.put("requestId", result.getRequestId());
                response.put("timestamp", result.getTimestamp().toString());
                response.put("status", result.getStatus().name());
                response.put("recognized", result.isRecognized());
                response.put("faceDetected", result.isFaceDetected());

                if (result.isRecognized()) {
                    RecognitionResult.MatchResult match = result.getBestMatch().get();
                    Map<String, Object> matchInfo = new LinkedHashMap<>();
                    matchInfo.put("identity", match.getIdentity().getName());
                    matchInfo.put("identityId", match.getIdentity().getId());
                    matchInfo.put("confidence", match.getConfidence());
                    matchInfo.put("distance", match.getDistance());
                    response.put("match", matchInfo);
                }

                if (!result.getAlternatives().isEmpty()) {
                    List<Map<String, Object>> alternatives = new ArrayList<>();
                    for (RecognitionResult.MatchResult alt : result.getTopAlternatives(5)) {
                        Map<String, Object> altInfo = new LinkedHashMap<>();
                        altInfo.put("identity", alt.getIdentity().getName());
                        altInfo.put("confidence", alt.getConfidence());
                        alternatives.add(altInfo);
                    }
                    response.put("alternatives", alternatives);
                }

                result.getMetrics().ifPresent(m -> {
                    Map<String, Object> metrics = new LinkedHashMap<>();
                    metrics.put("totalMs", m.getTotalTimeMs());
                    metrics.put("detectionMs", m.getDetectionTimeMs());
                    metrics.put("extractionMs", m.getExtractionTimeMs());
                    metrics.put("matchingMs", m.getMatchingTimeMs());
                    response.put("metrics", metrics);
                });

                sendJson(exchange, 200, response);
                logRequest(exchange, 200, System.currentTimeMillis() - start);

            } catch (Exception e) {
                sendError(exchange, 500, "Recognition error: " + e.getMessage());
                logRequest(exchange, 500, System.currentTimeMillis() - start);
            }
        }
    }

    private class EnrollHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            long start = System.currentTimeMillis();
            requestCounter.incrementAndGet();

            if (!checkAuth(exchange)) return;

            if (exchange.getRequestMethod().equals("OPTIONS")) {
                addCorsHeaders(exchange);
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            if (!exchange.getRequestMethod().equals("POST")) {
                sendError(exchange, 405, "Method not allowed. Use POST.");
                logRequest(exchange, 405, System.currentTimeMillis() - start);
                return;
            }

            // Get identity from query parameter
            String query = exchange.getRequestURI().getQuery();
            String identityName = null;
            if (query != null) {
                for (String param : query.split("&")) {
                    String[] kv = param.split("=");
                    if (kv.length == 2 && kv[0].equals("identity")) {
                        identityName = java.net.URLDecoder.decode(kv[1], "UTF-8");
                    }
                }
            }

            if (identityName == null || identityName.isEmpty()) {
                sendError(exchange, 400, "Missing 'identity' query parameter");
                logRequest(exchange, 400, System.currentTimeMillis() - start);
                return;
            }

            try {
                BufferedImage image = ImageIO.read(exchange.getRequestBody());
                if (image == null) {
                    sendError(exchange, 400, "Invalid or missing image data");
                    logRequest(exchange, 400, System.currentTimeMillis() - start);
                    return;
                }

                FaceImage faceImage = FaceImage.fromBufferedImage(image);
                Identity identity = service.enroll(faceImage, identityName);

                // Re-train if needed
                if (service.getIdentityCount() > 0) {
                    service.train();
                }

                Map<String, Object> response = new LinkedHashMap<>();
                response.put("success", true);
                response.put("message", "Face enrolled successfully");
                response.put("identityId", identity.getId());
                response.put("identityName", identity.getName());
                response.put("sampleCount", identity.getSampleCount());

                sendJson(exchange, 201, response);
                logRequest(exchange, 201, System.currentTimeMillis() - start);

            } catch (Exception e) {
                sendError(exchange, 500, "Enrollment error: " + e.getMessage());
                logRequest(exchange, 500, System.currentTimeMillis() - start);
            }
        }
    }

    private class IdentitiesHandler implements HttpHandler {
        @Override
        public void handle(HttpExchange exchange) throws IOException {
            long start = System.currentTimeMillis();
            requestCounter.incrementAndGet();

            if (!checkAuth(exchange)) return;

            if (exchange.getRequestMethod().equals("OPTIONS")) {
                addCorsHeaders(exchange);
                exchange.sendResponseHeaders(204, -1);
                return;
            }

            if (!exchange.getRequestMethod().equals("GET")) {
                sendError(exchange, 405, "Method not allowed. Use GET.");
                logRequest(exchange, 405, System.currentTimeMillis() - start);
                return;
            }

            List<Map<String, Object>> identities = new ArrayList<>();
            for (Identity id : service.getIdentities()) {
                Map<String, Object> info = new LinkedHashMap<>();
                info.put("id", id.getId());
                info.put("name", id.getName());
                info.put("sampleCount", id.getSampleCount());
                info.put("active", id.isActive());
                info.put("createdAt", id.getCreatedAt().toString());
                info.put("updatedAt", id.getUpdatedAt().toString());
                if (id.getExternalId() != null) {
                    info.put("externalId", id.getExternalId());
                }
                identities.add(info);
            }

            Map<String, Object> response = new LinkedHashMap<>();
            response.put("count", identities.size());
            response.put("identities", identities);

            sendJson(exchange, 200, response);
            logRequest(exchange, 200, System.currentTimeMillis() - start);
        }
    }
}
