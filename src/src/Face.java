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

public class Face {
    
    public File file;
    public Picture picture;
    public String classification;
    public String description;
    int globalwindowsize=5;
    
    public Face(File f) throws MalformedURLException {
        classification = null;
        description = "";
        file = f;
        
        load(false);
      
    }
    //manickam run the file
    public void load(boolean crop) throws MalformedURLException{
    	//System.out.println(file);
        Image im = new ImageIcon(file.toURL()).getImage();
        BufferedImage imb=new BufferedImage(im.getWidth(null),im.getHeight(null),BufferedImage.TYPE_INT_ARGB);
        
        Graphics2D g2d = imb.createGraphics();
        g2d.drawImage(im,0,0,null);
        g2d.dispose();
        
        if (crop) imb=recognise(recognise(imb));
        
        BufferedImage rim = this.resizeImage(imb, Main.IDEAL_IMAGE_SIZE, true);

        picture = new Picture(rim);
    }
    

    public BufferedImage recognise(BufferedImage im) {
        Calendar c=Calendar.getInstance();
        int height=im.getHeight(null);
        int width=im.getWidth(null);
        int scale=((height+width)/320);
        int leftcrop=width/2;
        int rightcrop=width/2;
        int topcrop=height/2;
        int bottomcrop=height/2;
        
        picture = new Picture(im);
        
        double[] originalpixelsd=picture.getImageColourPixels().clone();
        
        int[] originalpixels=new int[originalpixelsd.length];
        for (int i=0;i<originalpixelsd.length;i++) {
            originalpixels[i]=(int)originalpixelsd[i];
        }

        
        double[] pixels = picture.getImageColourPixels().clone();
        double[] pixels2 = pixels.clone();
        
        for (int i=2;i<width-2;i++) {
            for (int j=2;j<height-2;j++) {
                Color colour=getColor((int)pixels[i+j*width]);
                float[] hsb=Color.RGBtoHSB(colour.getRed(),colour.getGreen(),colour.getBlue(),null);
                
                double I=(Math.log(colour.getRed())+Math.log(colour.getGreen())+Math.log(colour.getBlue()))/3;
                double Rg=Math.log(colour.getRed())-Math.log(colour.getGreen());
                double By=Math.log(colour.getBlue())-(Math.log(colour.getBlue())+Math.log(colour.getRed()))/2;

                double hue = Math.atan2(Rg,By) * (180 / Math.PI);
                if (I <= 6 && (hue >= 70 && hue <= 150)) { // originally 30-130
                     //skin
                    pixels2[i+width*j]=colour.getRGB();
                }
                else {
                    for (int x=i-2;x<=i+2;x++) {
                        for (int y=j-2;y<=j+2;y++) {
                            pixels2[x+width*y]=Color.BLACK.getRGB();
                        }
                    }
                }
            }
        }
        
        int[] intpixels=new int[pixels2.length];
        for (int i=0;i<intpixels.length;i++) {
            intpixels[i]=(int)pixels2[i];
        }

     //   picture.display(intpixels,width,height);
        
        medianFilter(pixels2,globalwindowsize);

        for (int i=0;i<width;i++) {
            for (int j=0;j<height;j++) {
                Color colour=getColor((int)pixels2[i+width*j]);
                if (colour.getRGB()>(Color.BLACK.getRGB()+50)) {
                //  System.out.println(i+" "+j);
                    if ((i)<leftcrop) leftcrop=i;
                    if ((i)>rightcrop) rightcrop=i; 
                    if ((j)<topcrop) topcrop=j;
                    if ((j)>bottomcrop) bottomcrop=j; 
                }
            }
        }
        
    //  System.out.println("width:"+width+" "+leftcrop+" "+rightcrop);
    //  System.out.println("height:"+height+" "+topcrop+" "+bottomcrop);
                
        /*if(twopasscycle) {
            p.loadImage();
        }
       for (int i=leftcrop;i<rightcrop;i++) {
            p.pixels[i+width*(topcrop)]=Color.RED.getRGB();
            p.pixels[i+width*(bottomcrop)]=Color.RED.getRGB();
        }
        for (int j=topcrop;j<bottomcrop;j++) {
            p.pixels[leftcrop+width*j]=Color.RED.getRGB();
            p.pixels[rightcrop+width*j]=Color.RED.getRGB();
        }*/
    

        //picture.cropAndDisplay(intpixels,width,height,leftcrop,rightcrop,topcrop,bottomcrop);
       picture.cropAndDisplay(originalpixels,width,height,leftcrop,rightcrop,topcrop,bottomcrop);

    //  p2.display(pixels,width,height);
    /*  p2.setImage(p2.image);
        
        p.display(p.pixels,width,height);
        p.setImage(p.image);
        repaint();
        Calendar c2=Calendar.getInstance();
        status.setText("Recognition finished. Time elapsed: "+(c2.getTimeInMillis()-c.getTimeInMillis())+"ms");
        //System.out.println("Recognition finished. Time elapsed: "+(c2.getTimeInMillis()-c.getTimeInMillis())+"ms");*/
        
        return picture.img;
    }
    
    public void medianFilter(double[] pixels,int windowsize) {
        int height=picture.getHeight();
        int width=picture.getWidth();
        
        for (int i=windowsize/2;i<width-windowsize/2;i++) {
        //    progress.setValue((i*100)/(width-windowsize));
        //    progress.repaint();
            for (int j=windowsize/2;j<height-windowsize/2;j++) {
                ArrayList numbers=new ArrayList();
                for (int l=-windowsize/2;l<=windowsize/2;l++) {
                    for (int k=-windowsize/2;k<=windowsize/2;k++) {
                        numbers.add(new Integer((int)pixels[(i+k)+width*(j+l)]));
                    //  System.out.println((i+k)+width*(j+l));
                    }
                }
                Collections.sort(numbers);
                pixels[i+width*j]=((Integer)numbers.get(numbers.size()/2)).intValue();
            }
        }
    }
    
    public Color getColor(int rgb) {
        int red   = (rgb & (255<<16)) >> 16;
        int green = (rgb & (255<< 8)) >>  8;
        int blue  = (rgb &  255);
        return new Color(red,green,blue);
    }
    
    private double[] removeNoise(double[] pixels) {
        
        
        return pixels;
    }



   
    private BufferedImage resizeImage(Image orig, Dimension box, boolean fitOutside) {
        BufferedImage scaledBI = new BufferedImage(box.width, box.height, BufferedImage.TYPE_INT_RGB);
        Graphics2D g = (Graphics2D)scaledBI.getGraphics();
        g.setBackground(Color.BLACK); // need to clear()
        g.setRenderingHint(RenderingHints.KEY_INTERPOLATION,
                RenderingHints.VALUE_INTERPOLATION_BILINEAR); 
                
        float xRatio = (float)box.width / (float)orig.getWidth(null);
        float yRatio = (float)box.height / (float)orig.getHeight(null);
        
        float ratio;
        if(fitOutside) ratio = Math.max(xRatio, yRatio);
        else ratio = Math.min(xRatio, yRatio);
        
        int wNew = (int)(orig.getWidth(null) * ratio);
        int hNew = (int)(orig.getHeight(null) * ratio);
        int x = (box.width - wNew) / 2;
        int y = (box.height - hNew) / 2;
        
        g.drawImage(orig, x, y, wNew, hNew, null);
        
        return scaledBI;
    }
    
    
    
}
