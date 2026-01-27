---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description
A clear and concise description of what the bug is.

## To Reproduce
Steps to reproduce the behavior:
1. Initialize with '...'
2. Call method '...'
3. Pass parameters '...'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
What actually happened.

## Code Example
```java
// Minimal code example that reproduces the issue
FaceRecognitionService service = FaceRecognitionService.builder()
    .extractor(new EigenfacesExtractor())
    .classifier(new KNNClassifier())
    .build();

// ... steps that lead to the bug
```

## Error Output
```
// Stack trace or error message
```

## Environment
- **Java Version**: [e.g., 11.0.15]
- **OS**: [e.g., Windows 11, Ubuntu 22.04]
- **Library Version**: [e.g., 2.0.0]
- **Maven Version**: [e.g., 3.8.6]

## Additional Context
Add any other context about the problem here.

## Possible Solution
If you have ideas on how to fix this, please describe them.
