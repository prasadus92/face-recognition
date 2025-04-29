package src;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ImageObserver;
import java.awt.image.PixelGrabber;

import javax.swing.JComponent;

/**
 * A component that displays and processes images for face recognition.
 * Provides methods for image manipulation and pixel extraction.
 *
 * @author Prasad Subrahmanya
 * @version 1.0
 */
public class Picture extends JComponent {
	
	private static final long serialVersionUID = 1L;
	private static final double COLOR_CHANNEL_WEIGHT = 3.0;
	
	private BufferedImage image;

	/**
	 * Creates a new Picture with the specified image.
	 *
	 * @param image the image to display
	 */
	public Picture(BufferedImage image) {
		this.image = image;
	}
	
	/**
	 * Gets the current image.
	 *
	 * @return The current image
	 */
	public BufferedImage getImage() {
		return image;
	}
	
	/**
	 * Sets the image to display.
	 *
	 * @param image The new image to display
	 */
	public void setImage(BufferedImage image) {
		this.image = image;
		repaint();
	}
	
	@Override
	public void paint(Graphics g) {
		g.drawImage(image, 0, 0, this);
	}
	
	/**
	 * Gets the grayscale pixel values of the image.
	 *
	 * @return an array of grayscale pixel values
	 */
	public double[] getImagePixels() {
		int width = image.getWidth(this);
		int height = image.getHeight(this);
		int[] pixels = new int[width * height];
		
		PixelGrabber pixelGrabber = new PixelGrabber(image, 0, 0, width, height, pixels, 0, width);
		try {
			pixelGrabber.grabPixels();
		} catch (InterruptedException e) {
			System.err.println("Interrupted while waiting for pixels: " + e.getMessage());
			return new double[0];
		}
		
		if ((pixelGrabber.getStatus() & ImageObserver.ABORT) != 0) {
			System.err.println("Image fetch aborted or errored");
			return new double[0];
		}

		double[] grayscalePixels = new double[width * height];
		ColorModel colorModel = pixelGrabber.getColorModel();
		
		for (int i = 0; i < grayscalePixels.length; i++) {
			grayscalePixels[i] = (colorModel.getBlue(pixels[i]) + 
								colorModel.getGreen(pixels[i]) + 
								colorModel.getRed(pixels[i])) / COLOR_CHANNEL_WEIGHT;
		}
		
		return grayscalePixels;
	}
    
	/**
	 * Gets the RGB pixel values of the image.
	 *
	 * @return an array of RGB pixel values
	 */
	public double[] getImageColorPixels() {
		int width = image.getWidth(this);
		int height = image.getHeight(this);
		int[] pixels = new int[width * height];
		
		PixelGrabber pixelGrabber = new PixelGrabber(image, 0, 0, width, height, pixels, 0, width);
		try {
			pixelGrabber.grabPixels();
		} catch (InterruptedException e) {
			System.err.println("Interrupted while waiting for pixels: " + e.getMessage());
			return new double[0];
		}
		
		if ((pixelGrabber.getStatus() & ImageObserver.ABORT) != 0) {
			System.err.println("Image fetch aborted or errored");
			return new double[0];
		}

		double[] colorPixels = new double[width * height];
		ColorModel colorModel = pixelGrabber.getColorModel();
		
		for (int i = 0; i < colorPixels.length; i++) {
			Color color = new Color(
				colorModel.getRed(pixels[i]),
				colorModel.getGreen(pixels[i]),
				colorModel.getBlue(pixels[i])
			);
			colorPixels[i] = color.getRGB();
		}
		
		return colorPixels;
	}
    
	/**
	 * Crops and displays a portion of the image.
	 *
	 * @param resultPixels the pixel data to display
	 * @param width the width of the image
	 * @param height the height of the image
	 * @param leftCrop the left crop boundary
	 * @param rightCrop the right crop boundary
	 * @param topCrop the top crop boundary
	 * @param bottomCrop the bottom crop boundary
	 */
	public void cropAndDisplay(int[] resultPixels, int width, int height,
							 int leftCrop, int rightCrop, int topCrop, int bottomCrop) {
		image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		image.setRGB(0, 0, width, height, resultPixels, 0, width);
		image = image.getSubimage(leftCrop, topCrop, rightCrop - leftCrop, bottomCrop - topCrop);
		repaint();
	}

	/**
	 * Displays the image with the specified pixel data.
	 *
	 * @param resultPixels the pixel data to display
	 * @param width the width of the image
	 * @param height the height of the image
	 */
	public void display(int[] resultPixels, int width, int height) {
		image = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
		image.setRGB(0, 0, width, height, resultPixels, 0, width);
		repaint();
	}
}
