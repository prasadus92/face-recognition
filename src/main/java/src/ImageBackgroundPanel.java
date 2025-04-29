package src;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import javax.imageio.*;
import java.io.*;

/**
 * A JPanel that displays a background image.
 * The image is scaled to fit the panel's dimensions.
 * Provides a customizable background for the application interface.
 *
 * @author Prasad Subrahmanya
 * @version 1.0
 * @since 1.0
 */
public class ImageBackgroundPanel extends JPanel {
    private static final long serialVersionUID = 1L;
    private static final String BACKGROUND_IMAGE_PATH = "/src/src/bkd.png";
    
    private BufferedImage backgroundImage;

    /**
     * Creates a new ImageBackgroundPanel with the default background image.
     * If the image cannot be loaded, a blank panel will be displayed.
     */
    public ImageBackgroundPanel() {
        loadBackgroundImage();
    }

    /**
     * Loads the background image from the specified path.
     * If the image cannot be loaded, a blank panel will be displayed.
     */
    private void loadBackgroundImage() {
        try {
            String imagePath = System.getProperty("user.dir").replace('\\', '/') + BACKGROUND_IMAGE_PATH;
            backgroundImage = ImageIO.read(new File(imagePath));
        } catch (IOException e) {
            System.err.println("Failed to load background image: " + e.getMessage());
        }
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        if (backgroundImage != null) {
            g.drawImage(backgroundImage, 0, 0, getWidth(), getHeight(), this);
        }
    }
}
