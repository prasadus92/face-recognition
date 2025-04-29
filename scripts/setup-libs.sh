#!/bin/bash

# Create lib directory if it doesn't exist
mkdir -p lib

# Download Java 3D libraries from NEA webstart and Maven
echo "Downloading Java 3D libraries..."
wget https://www.oecd-nea.org/webstart/java3d-1.5.2/j3d/1.5.2/j3dcore.jar -O lib/j3d-core.jar
wget https://www.oecd-nea.org/webstart/java3d-1.5.2/j3d/1.5.2/j3dutils.jar -O lib/j3d-utils.jar
wget https://repo1.maven.org/maven2/javax/vecmath/vecmath/1.5.2/vecmath-1.5.2.jar -O lib/vecmath.jar

# Verify downloads
for jar in lib/j3d-core.jar lib/j3d-utils.jar lib/vecmath.jar; do
    if [ ! -s "$jar" ]; then
        echo "Error: $jar is empty or missing"
        exit 1
    fi
done

# Download JAI libraries
echo "Downloading JAI libraries..."
wget https://download.java.net/media/jai/builds/release/1_1_3/jai-1_1_3-lib-linux-amd64.tar.gz
tar -xzf jai-1_1_3-lib-linux-amd64.tar.gz
cp jai-1_1_3/lib/*.jar lib/
rm -rf jai-1_1_3 jai-1_1_3-lib-linux-amd64.tar.gz

# Verify all downloads
for jar in lib/*.jar; do
    if [ ! -s "$jar" ]; then
        echo "Error: $jar is empty or missing"
        exit 1
    fi
done

echo "Library setup complete!" 