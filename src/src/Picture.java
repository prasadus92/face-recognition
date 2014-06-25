package src;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ImageObserver;
import java.awt.image.PixelGrabber;

import javax.swing.JComponent;

public class Picture extends JComponent {
	
	private static final long serialVersionUID = 1L;
	
	BufferedImage img;

	
	public Picture(BufferedImage i) {
		img = i;
	}
	
	public void setImage(BufferedImage i) {
		img = i;
	}
	
	public void paint(Graphics g) {
		g.drawImage(img, 0, 0, this);
	}
	
	public double[] getImagePixels() {
		
		int w = img.getWidth(this);
		int h = img.getHeight(this);
		int[] pixels = new int[w * h];
		PixelGrabber pg = new PixelGrabber(img, 0, 0, w, h, pixels, 0, w);
		try {
			pg.grabPixels();
		} catch (InterruptedException e) {
			System.err.println("interrupted waiting for pixels!");
			return new double[0];
		}
		if ((pg.getStatus() & ImageObserver.ABORT) != 0) {
			System.err.println("image fetch aborted or errored");
			return new double[0];
		}
		double[] ret =new double[w*h];
		ColorModel cm = pg.getColorModel();
		for (int i=0; i<ret.length; i++)
		{
			ret[i] = cm.getBlue(pixels[i]) + cm.getGreen(pixels[i]) + cm.getRed(pixels[i]);
			ret[i] /= 3.0;
		}
		return ret;
	}
    
	public double[] getImageColourPixels() {
        
        int w = img.getWidth(this);
        int h = img.getHeight(this);
        int[] pixels = new int[w * h];
        PixelGrabber pg = new PixelGrabber(img, 0, 0, w, h, pixels, 0, w);
        try {
            pg.grabPixels();
        } catch (InterruptedException e) {
            System.err.println("interrupted waiting for pixels!");
            return new double[0];
        }
        if ((pg.getStatus() & ImageObserver.ABORT) != 0) {
            System.err.println("image fetch aborted or errored");
            return new double[0];
        }
        double[] ret =new double[w*h];
        ColorModel cm = pg.getColorModel();
        for (int i=0; i<ret.length; i++)
        {
            Color c=new Color(cm.getRed(pixels[i]),cm.getGreen(pixels[i]),cm.getBlue(pixels[i]));
            ret[i]=c.getRGB();
         
        }
        return ret;
    }
    
    public void cropAndDisplay(int[] resultpixels,int w, int h,int leftcrop,int rightcrop,int topcrop,int bottomcrop) {
        img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        img.setRGB(0, 0, w, h, resultpixels, 0, w);
        img = img.getSubimage(leftcrop,topcrop,(rightcrop-leftcrop),(bottomcrop-topcrop));
    
    }

	
	public void display(int[] resultpixels, int w, int h) {
		img = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
		img.setRGB(0, 0, w, h, resultpixels, 0, w);
	
	}
}
