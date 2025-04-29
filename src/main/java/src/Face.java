/* Author: Prasad U S
*
*
*/
package src;

import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics2D;
import java.awt.Image;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.io.File;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Collections;

import javax.swing.ImageIcon;

/**
 * Represents a face image in the face recognition system.
 * Provides methods for face detection, image processing, and feature extraction.
 *
 * @author Prasad Subrahmanya
 * @version 1.0
 * @since 1.0
 */
public class Face {
    private static final int WINDOW_SIZE = 5;
    private static final double SKIN_COLOR_INTENSITY_THRESHOLD = 6.0;
    private static final double SKIN_COLOR_HUE_MIN = 70.0;
    private static final double SKIN_COLOR_HUE_MAX = 150.0;
    private static final int BLACK_COLOR_THRESHOLD = 50;
    private static final int SCALE_FACTOR = 320;
    private static final int BORDER_MARGIN = 2;
    private static final Dimension IDEAL_IMAGE_SIZE = new Dimension(48, 64);
    private static final long serialVersionUID = 1L;

    private File file;
    private Picture picture;
    private String classification;
    private String description;

    /**
     * Creates a new Face instance from a file.
     *
     * @param file the image file containing the face
     * @throws MalformedURLException if the file URL is malformed
     */
    public Face(File file) throws MalformedURLException {
        this.classification = null;
        this.description = "";
        this.file = file;
        load(false);
    }

    /**
     * Gets the file associated with this face.
     *
     * @return the file
     */
    public File getFile() {
        return file;
    }

    /**
     * Gets the picture associated with this face.
     *
     * @return the picture
     */
    public Picture getPicture() {
        return picture;
    }

    /**
     * Gets the classification of this face.
     *
     * @return the classification
     */
    public String getClassification() {
        return classification;
    }

    /**
     * Sets the classification of this face.
     *
     * @param classification the classification to set
     */
    public void setClassification(String classification) {
        this.classification = classification;
    }

    /**
     * Gets the description of this face.
     *
     * @return the description
     */
    public String getDescription() {
        return description;
    }

    /**
     * Sets the description of this face.
     *
     * @param description the description to set
     */
    public void setDescription(String description) {
        this.description = description;
    }

    /**
     * Loads the face image from the file.
     *
     * @param crop whether to crop the image to detect the face
     * @throws MalformedURLException if the file URL is malformed
     */
    public void load(boolean crop) throws MalformedURLException {
        Image image = new ImageIcon(file.toURL()).getImage();
        BufferedImage bufferedImage = new BufferedImage(
            image.getWidth(null),
            image.getHeight(null),
            BufferedImage.TYPE_INT_ARGB
        );
        
        Graphics2D graphics = bufferedImage.createGraphics();
        graphics.drawImage(image, 0, 0, null);
        graphics.dispose();
        
        if (crop) {
            bufferedImage = recognize(recognize(bufferedImage));
        }
        
        BufferedImage resizedImage = resizeImage(bufferedImage, IDEAL_IMAGE_SIZE, true);
        picture = new Picture(resizedImage);
    }

    /**
     * Recognizes and extracts the face region from the image.
     *
     * @param image the input image
     * @return the processed image with the face region
     */
    public BufferedImage recognize(BufferedImage image) {
        int height = image.getHeight(null);
        int width = image.getWidth(null);
        int scale = ((height + width) / SCALE_FACTOR);
        int leftCrop = width / 2;
        int rightCrop = width / 2;
        int topCrop = height / 2;
        int bottomCrop = height / 2;
        
        picture = new Picture(image);
        
        double[] originalPixels = picture.getImageColorPixels().clone();
        int[] originalPixelsInt = new int[originalPixels.length];
        for (int i = 0; i < originalPixels.length; i++) {
            originalPixelsInt[i] = (int) originalPixels[i];
        }

        double[] pixels = picture.getImageColorPixels().clone();
        double[] processedPixels = pixels.clone();
        
        processSkinColorDetection(width, height, pixels, processedPixels);
        detectFaceBoundaries(width, height, processedPixels, leftCrop, rightCrop, topCrop, bottomCrop);
        
        picture.cropAndDisplay(originalPixelsInt, width, height, leftCrop, rightCrop, topCrop, bottomCrop);
        
        return picture.getImage();
    }

    /**
     * Processes the image to detect skin color regions.
     */
    private void processSkinColorDetection(int width, int height, double[] pixels, double[] processedPixels) {
        for (int i = BORDER_MARGIN; i < width - BORDER_MARGIN; i++) {
            for (int j = BORDER_MARGIN; j < height - BORDER_MARGIN; j++) {
                Color color = getColor((int) pixels[i + j * width]);
                float[] hsb = Color.RGBtoHSB(color.getRed(), color.getGreen(), color.getBlue(), null);
                
                double intensity = (Math.log(color.getRed()) + Math.log(color.getGreen()) + Math.log(color.getBlue())) / 3;
                double redGreen = Math.log(color.getRed()) - Math.log(color.getGreen());
                double blueYellow = Math.log(color.getBlue()) - (Math.log(color.getBlue()) + Math.log(color.getRed())) / 2;

                double hue = Math.atan2(redGreen, blueYellow) * (180 / Math.PI);
                
                if (intensity <= SKIN_COLOR_INTENSITY_THRESHOLD && (hue >= SKIN_COLOR_HUE_MIN && hue <= SKIN_COLOR_HUE_MAX)) {
                    processedPixels[i + width * j] = color.getRGB();
                } else {
                    clearNeighborhood(processedPixels, width, i, j);
                }
            }
        }
    }

    /**
     * Clears the neighborhood of a pixel by setting it to black.
     */
    private void clearNeighborhood(double[] pixels, int width, int x, int y) {
        for (int i = x - BORDER_MARGIN; i <= x + BORDER_MARGIN; i++) {
            for (int j = y - BORDER_MARGIN; j <= y + BORDER_MARGIN; j++) {
                pixels[i + width * j] = Color.BLACK.getRGB();
            }
        }
    }

    /**
     * Detects the boundaries of the face in the image.
     */
    private void detectFaceBoundaries(int width, int height, double[] pixels, 
                                    int leftCrop, int rightCrop, int topCrop, int bottomCrop) {
        for (int i = 0; i < width; i++) {
            for (int j = 0; j < height; j++) {
                Color color = getColor((int) pixels[i + width * j]);
                if (color.getRGB() > (Color.BLACK.getRGB() + BLACK_COLOR_THRESHOLD)) {
                    if (i < leftCrop) leftCrop = i;
                    if (i > rightCrop) rightCrop = i;
                    if (j < topCrop) topCrop = j;
                    if (j > bottomCrop) bottomCrop = j;
                }
            }
        }
    }

    /**
     * Applies a median filter to the image.
     *
     * @param pixels the pixel array to filter
     * @param windowSize the size of the filter window
     */
    public void medianFilter(double[] pixels, int windowSize) {
        int height = picture.getHeight();
        int width = picture.getWidth();
        
        for (int i = windowSize / 2; i < width - windowSize / 2; i++) {
            for (int j = windowSize / 2; j < height - windowSize / 2; j++) {
                ArrayList<Integer> numbers = new ArrayList<>();
                for (int l = -windowSize / 2; l <= windowSize / 2; l++) {
                    for (int k = -windowSize / 2; k <= windowSize / 2; k++) {
                        numbers.add((int) pixels[(i + k) + width * (j + l)]);
                    }
                }
                Collections.sort(numbers);
                pixels[i + width * j] = numbers.get(numbers.size() / 2);
            }
        }
    }

    /**
     * Extracts RGB components from an integer color value.
     *
     * @param rgb the RGB color value
     * @return the Color object
     */
    public Color getColor(int rgb) {
        int red = (rgb & (255 << 16)) >> 16;
        int green = (rgb & (255 << 8)) >> 8;
        int blue = (rgb & 255);
        return new Color(red, green, blue);
    }

    /**
     * Resizes an image to fit within the specified dimensions.
     *
     * @param original the original image
     * @param box the target dimensions
     * @param fitOutside whether to fit outside the box
     * @return the resized image
     */
    private BufferedImage resizeImage(Image original, Dimension box, boolean fitOutside) {
        BufferedImage scaledImage = new BufferedImage(box.width, box.height, BufferedImage.TYPE_INT_RGB);
        Graphics2D graphics = scaledImage.createGraphics();
        graphics.setBackground(Color.BLACK);
        graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
        
        float xRatio = (float) box.width / (float) original.getWidth(null);
        float yRatio = (float) box.height / (float) original.getHeight(null);
        float ratio = fitOutside ? Math.max(xRatio, yRatio) : Math.min(xRatio, yRatio);
        
        int newWidth = Math.round(original.getWidth(null) * ratio);
        int newHeight = Math.round(original.getHeight(null) * ratio);
        
        graphics.drawImage(original, 0, 0, newWidth, newHeight, null);
        graphics.dispose();
        
        return scaledImage;
    }
}
