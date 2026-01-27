# Contributing to Face Recognition System

Thank you for your interest in contributing to the Face Recognition System! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

---

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please:

- Be respectful and considerate in all interactions
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Accept constructive criticism gracefully
- Show empathy towards other community members

---

## Getting Started

### Prerequisites

- Java Development Kit (JDK) 8 or higher
- Maven 3.6 or higher
- Git
- IDE (IntelliJ IDEA, Eclipse, or VS Code recommended)

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/face-recognition.git
   cd face-recognition
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/prasadus92/face-recognition.git
   ```

---

## Development Setup

### Building the Project

```bash
# Install dependencies and build
mvn clean install

# Build without running tests
mvn clean install -DskipTests
```

### Running Tests

```bash
# Run all tests
mvn test

# Run specific test class
mvn test -Dtest=EigenfacesExtractorTest

# Run tests with coverage report
mvn clean test jacoco:report
# View report at target/site/jacoco/index.html
```

### IDE Setup

**IntelliJ IDEA:**
1. Open the project folder
2. Import as Maven project
3. Enable annotation processing if prompted

**Eclipse:**
1. File -> Import -> Maven -> Existing Maven Projects
2. Select the project folder
3. Import

---

## Making Changes

### Branching Strategy

We use a simplified Git Flow:

- `main` - Production-ready code
- `develop` - Integration branch for features
- `feature/*` - New features
- `bugfix/*` - Bug fixes
- `hotfix/*` - Critical production fixes

### Creating a Feature Branch

```bash
# Sync with upstream
git fetch upstream
git checkout develop
git merge upstream/develop

# Create feature branch
git checkout -b feature/your-feature-name
```

### Commit Messages

Follow the conventional commits format:

```
type(scope): subject

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style (formatting, semicolons, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Build process, dependencies, etc.

**Examples:**
```
feat(extractor): add Fisherfaces algorithm implementation

fix(classifier): correct distance calculation in KNN

docs(readme): update installation instructions
```

---

## Coding Standards

### Java Style Guide

We follow Google Java Style Guide with these key points:

#### Formatting
- Indentation: 4 spaces (no tabs)
- Line length: 120 characters maximum
- Braces: K&R style (opening brace on same line)

```java
// Good
public void doSomething() {
    if (condition) {
        // code
    }
}

// Bad
public void doSomething()
{
    if (condition)
    {
        // code
    }
}
```

#### Naming Conventions
- Classes: `PascalCase` (e.g., `FaceRecognitionService`)
- Methods: `camelCase` (e.g., `extractFeatures`)
- Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_DIMENSION`)
- Variables: `camelCase` (e.g., `featureVector`)
- Packages: `lowercase` (e.g., `com.facerecognition.domain`)

#### Documentation
- All public classes must have Javadoc
- All public methods must have Javadoc
- Use `@param`, `@return`, `@throws` tags appropriately

```java
/**
 * Extracts features from a face image.
 *
 * <p>This method normalizes the input image and projects it
 * onto the eigenface space.</p>
 *
 * @param face the face image to process
 * @return the extracted feature vector
 * @throws IllegalStateException if extractor is not trained
 */
public FeatureVector extract(FaceImage face) {
    // implementation
}
```

### Architecture Guidelines

Follow Clean Architecture principles:

1. **Domain Layer** - Core business logic (no external dependencies)
   - Entities, value objects
   - Domain services (interfaces)
   - Repository interfaces

2. **Application Layer** - Application-specific logic
   - Use cases / application services
   - DTOs for input/output

3. **Infrastructure Layer** - External implementations
   - Algorithm implementations
   - Database repositories
   - External service adapters

4. **API Layer** - User interfaces
   - REST controllers
   - CLI handlers
   - GUI components

**Dependency Rule:** Inner layers cannot depend on outer layers.

---

## Testing Guidelines

### Test Structure

```
src/test/java/
├── com/facerecognition/
│   ├── unit/              # Unit tests
│   │   ├── domain/        # Domain model tests
│   │   ├── application/   # Service tests
│   │   └── infrastructure/# Algorithm tests
│   ├── integration/       # Integration tests
│   └── benchmark/         # Performance tests
```

### Writing Unit Tests

Use JUnit 5 with these conventions:

```java
@DisplayName("EigenfacesExtractor Tests")
class EigenfacesExtractorTest {

    @Nested
    @DisplayName("Training")
    class TrainingTests {

        @Test
        @DisplayName("Trains successfully with valid data")
        void trainsSuccessfully() {
            // Arrange
            List<FaceImage> faces = createTestFaces();

            // Act
            extractor.train(faces, null);

            // Assert
            assertTrue(extractor.isTrained());
        }

        @Test
        @DisplayName("Throws on empty training set")
        void throwsOnEmptyTrainingSet() {
            assertThrows(IllegalArgumentException.class,
                () -> extractor.train(new ArrayList<>(), null));
        }
    }
}
```

### Test Coverage

- Aim for 80%+ code coverage
- 100% coverage for public APIs
- Focus on meaningful tests, not just coverage numbers

### Test Categories

Use these annotations for test categorization:

```java
@Tag("unit")           // Unit tests (fast, isolated)
@Tag("integration")    // Integration tests
@Tag("slow")          // Tests that take > 1 second
@Tag("benchmark")     // Performance benchmarks
```

---

## Submitting Changes

### Before Submitting

1. Ensure all tests pass:
   ```bash
   mvn clean test
   ```

2. Check code style:
   ```bash
   mvn checkstyle:check
   ```

3. Update documentation if needed

4. Add/update tests for your changes

### Creating a Pull Request

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

2. Go to GitHub and create a Pull Request

3. Fill out the PR template:
   - Clear description of changes
   - Link to related issue(s)
   - Testing done
   - Screenshots (if UI changes)

### PR Title Format

```
type(scope): brief description
```

Example: `feat(extractor): add LBPH feature extractor`

---

## Review Process

### What We Look For

- Code quality and style adherence
- Test coverage and quality
- Documentation completeness
- Performance implications
- Security considerations
- Backward compatibility

### Review Timeline

- Initial review: 2-3 business days
- Follow-up reviews: 1-2 business days

### After Review

- Address all comments
- Push updates to the same branch
- Mark resolved conversations as resolved
- Request re-review when ready

---

## Areas for Contribution

### Good First Issues

Look for issues labeled `good first issue` for starter tasks.

### Wanted Features

- Deep learning integration (ONNX Runtime)
- Additional face detection methods
- Real-time video processing
- Web interface
- Additional distance metrics
- Performance optimizations

### Documentation

- API documentation improvements
- Tutorial creation
- Example projects
- Translations

---

## Questions?

- Open a Discussion on GitHub
- Tag maintainers in issues
- Email: prasadus92@gmail.com

---

Thank you for contributing to Face Recognition System!
