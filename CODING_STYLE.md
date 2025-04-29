# Coding Style Guide

## Java Code Style

### General Rules
- Use 4 spaces for indentation (no tabs)
- Maximum line length: 120 characters
- Use UTF-8 encoding
- Use Unix-style line endings (LF)

### Naming Conventions
- Class names: PascalCase (e.g., `FeatureVector`)
- Method names: camelCase (e.g., `getFeatureVector`)
- Variable names: camelCase (e.g., `imageWidth`)
- Constants: UPPER_SNAKE_CASE (e.g., `MAX_WIDTH`)
- Package names: lowercase (e.g., `src`)

### File Organization
- One public class per file
- File name should match the public class name
- Order of declarations:
  1. Constants
  2. Static fields
  3. Instance fields
  4. Constructors
  5. Methods

### Documentation
- Use Javadoc for all public classes and methods
- Include `@author`, `@version`, `@param`, `@return`, and `@throws` tags where appropriate
- Keep documentation up to date with code changes

### Code Formatting
- Braces on new lines for class and method declarations
- Braces on same line for control structures
- One statement per line
- One blank line between methods
- Two blank lines between classes

### Best Practices
- Use meaningful variable names
- Keep methods small and focused
- Use constants for magic numbers
- Handle exceptions appropriately
- Use proper access modifiers
- Follow SOLID principles
- Write unit tests for new functionality

## Author Information
- Author: Prasad Subrahmanya
- Email: [Your Email]
- GitHub: [Your GitHub]

## Version Control
- Use meaningful commit messages
- Follow Git Flow branching model
- Keep commits atomic and focused

## Build and Dependencies
- Use Maven for dependency management
- Keep dependencies up to date
- Document all external dependencies

## Testing
- Write unit tests for all new functionality
- Maintain test coverage above 80%
- Use JUnit for testing
- Follow TDD practices where possible 