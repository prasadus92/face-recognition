/* Author: Prasad U S
 * 
 * 
 */
package src;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import javax.imageio.*;
import java.io.*;
import java.net.URL;


public class ImageBackgroundPanel extends JPanel {

    BufferedImage img;
    public String imgPath;

    public ImageBackgroundPanel() {
        try {
            
            imgPath=System.getProperty("user.dir");
            imgPath=imgPath.replace('\\','/');
            img = ImageIO.read(new File(imgPath + "/src/src/bkd.png"));

        } catch (IOException e) {
            System.out.println("Image face.png missing\n" + e);
        }
    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        g.drawImage(img, 0, 0, getWidth(), getHeight(), this);
    }
}
