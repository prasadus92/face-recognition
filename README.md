# Face Recognition System

[![Build Status](https://github.com/prasadus92/face-recognition/actions/workflows/build.yml/badge.svg)](https://github.com/prasadus92/face-recognition/actions/workflows/build.yml)

## Author
**Prasad Subrahmanya**  
[GitHub Profile](https://github.com/prasadus92)

A Java-based face recognition system that uses eigenfaces for face recognition under various conditions including expressions, occlusions, and pose variations.

## Features

- Face recognition under different conditions:
  - Facial expressions
  - Occlusions (e.g., glasses)
  - Pose variations (up to 60 degrees)
- Interactive GUI for:
  - Loading and processing face images
  - Training the recognition system
  - Testing face recognition
  - Visualizing results
- Support for batch processing
- Results visualization with 3D charts

## Technical Details

- Uses eigenfaces for face recognition
- Implements Two-Stage Classification and Detection (TSCD)
- Supports 3D face visualization
- Uses Java 3D for 3D rendering
- MySQL database for user management

## Prerequisites

- Java Development Kit (JDK) 8 or higher
- Maven 3.6 or higher
- MySQL 5.7 or higher
- Java 3D API

## Building the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/prasadus92/face-recognition.git
   cd face-recognition
   ```

2. Install Java 3D API:
   - Download Java 3D API from Oracle's website
   - Install it on your system
   - Add the Java 3D libraries to your system's Java library path

3. Set up MySQL:
   ```sql
   CREATE DATABASE face_recognition;
   USE face_recognition;
   CREATE TABLE users (
       UserName VARCHAR(30),
       Email VARCHAR(50),
       Phone VARCHAR(10),
       Password VARCHAR(20)
   );
   ```

4. Install JAI libraries:
   - Download JAI libraries from Oracle's website
   - Place the following files in the `lib` directory:
     - `jai_core.jar`
     - `jai_codec.jar`
     - `mlibwrapper_jai.jar`

5. Build the project:
   ```bash
   mvn clean install
   ```

## Running the Application

1. Start MySQL server

2. Run the application:
   ```bash
   mvn exec:java -Dexec.mainClass="src.Main"
   ```

## Platform-Specific Setup

### Windows
1. Install Microsoft Visual C++ Redistributable Package (vcredist_x86.exe)
2. Copy the DLLs from `platform-specific/windows/` to your system's Java library path:
   ```bash
   copy platform-specific\windows\*.dll %JAVA_HOME%\bin
   ```

### Linux
1. Install required system libraries:
   ```bash
   sudo apt-get install libj3d-java
   ```

### macOS
1. Install required system libraries:
   ```bash
   brew install java3d
   ```

## Project Structure

- `src/main/java/src/` - Source code
  - `Main.java` - Main application entry point
  - `Face.java` - Face recognition core logic
  - `FeatureSpace.java` - Feature extraction and analysis
  - `TSCD.java` - Two-Stage Classification and Detection
  - `FaceBrowser.java` - Face visualization component
- `src/main/resources/` - Application resources
  - `face.png` - Application icon
  - `bkd.png` - Background image
- `lib/` - Local dependencies
  - JAI libraries
- `platform-specific/` - Platform-specific files
  - `windows/` - Windows-specific DLLs
- `pom.xml` - Maven project configuration
- `README.md` - Project documentation

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [License.txt](License.txt) file for details.

## Acknowledgments

- Bosphorus Database for testing data
- Java 3D API team
- All contributors to the project
