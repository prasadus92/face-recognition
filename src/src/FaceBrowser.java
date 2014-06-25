package src;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Component;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashMap;

import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JComponent;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.border.TitledBorder;

public class FaceBrowser extends JPanel {

    private static final long serialVersionUID = 1L;
    private ArrayList<FaceItem> m_faces;
    private int height = 0;
    private HashMap<FaceItem, Face> mapItem2Face = new HashMap<FaceItem, Face>();
    private HashMap<Face, FaceItem> mapFace2Item = new HashMap<Face, FaceItem>();

    public FaceBrowser() {
        m_faces = new ArrayList<FaceItem>();
        this.setPreferredSize(new Dimension(200, 500));
        this.setBackground(this.getBackground().brighter());
    }

    public void refresh() {
        for (FaceItem fi : mapItem2Face.keySet()) {
            fi.refresh();
        }
    }

    public void addFace(Face f) {
        FaceItem fi = new FaceItem(f);
        this.add(fi);
        mapItem2Face.put(fi, f);
        mapFace2Item.put(f, fi);
    }

    public void empty() {
        this.removeAll();
        m_faces.clear();
        mapItem2Face.clear();
        mapFace2Item.clear();
        doLayout();
    }

    @Override
    public Dimension getMinimumSize() {
        return new Dimension(256, height);
    }

    @Override
    public Dimension getPreferredSize() {
        return getMinimumSize();
    }

    public void highlightClassifiedAs(String s) {
        for (FaceItem fi : mapItem2Face.keySet()) {
            Face face = fi.face;
            if (face != null && face.classification != null) {
                boolean sameGroup = face.classification.equals(s);
                fi.setHighlighted(sameGroup);
            }
        }
    }

    public void orderAs(FeatureSpace.fd_pair[] facePairs) {
        this.removeAll();

        for (FeatureSpace.fd_pair fd : facePairs) {
            if (fd.face.classification.equals(Main.classification)) {
                FaceItem fi = mapFace2Item.get(fd.face);
                fi.setFace(fd.face);
                fi.setDistance(fd.dist);
                this.add(fi);
            }
        }

    }

    @Override
    public void doLayout() {
        // TODO Auto-generated method stub
        super.doLayout();

        Component[] components = this.getComponents();
        int cury = 0;
        for (Component c : components) {
            c.setLocation(0, cury);
            c.setSize(this.getWidth(), c.getHeight());
            cury += c.getHeight();
        }

        height = cury;

        this.revalidate();
    }
}

class FaceItem extends JPanel {

    private static final long serialVersionUID = 1L;
    Face face;
    ImageIcon image;
    JLabel jlImage;
    JLabel jlText;
    TitledBorder border;
    ImageIcon imageEigen;
    JLabel jlEigenface;
    double dist = -1;

    public FaceItem() {
        init();
    }

    public void setDistance(double dist) {
        this.dist = dist;

        updateLabel();

        double amt = dist / 4096;
        if (amt > 1) {
            amt = 1;
        }
        amt = 0.5 + amt / 2;
        this.setBackground(new Color((float) amt, 1.0f, (float) amt));
        this.setOpaque(true);


    }

    private void updateLabel() {
        String text = "<html>";

        text += "<font size=+1><font color=#7f7f7f>Classification:</font> ";
        if (this.face.classification == null) {
            text += "<font color=#7f0000>[unclassified]</font>";
        } else {
            text += "<font color=#00007f>" + this.face.classification + "</font>";
        }
        text += "</b></font>";

        if (this.dist >= 0) {
            text += ("<br><b>" + "Distance: " + this.dist + "</b>");

        }

        text += "<br>" + this.face.description + "";

        text += "<br><font size=-2 color=#7f7f7f>" + this.face.file.getAbsolutePath() + "</font>";



        text += "</html>";
        jlText.setText(text);
    }

    public void setHighlighted(boolean b) {
        this.setOpaque(b);
        if (b) {
            border.setTitleColor(Color.BLACK);
            border.setBorder(BorderFactory.createLineBorder(Color.BLACK));
        } else {
            border.setTitleColor(Color.GRAY);
            border.setBorder(BorderFactory.createLineBorder(Color.GRAY));
        }
    }

    public void refresh() {
        this.image = new ImageIcon(this.face.picture.img);
        jlImage.setIcon(this.image);
    }

    public void setFace(Face f) {
        this.face = f;

        refresh();

        border.setTitle(f.file.getName());


        updateLabel();



        Insets i = jlImage.getInsets();
        jlImage.setPreferredSize(
                new Dimension(
                image.getIconWidth() + i.left + i.right,
                image.getIconHeight() + i.top + i.bottom));


    }

    private void init() {
        BorderLayout layout = new BorderLayout();
        this.setLayout(layout);

        border = BorderFactory.createTitledBorder(BorderFactory.createLineBorder(this.getBackground()), "");
        //border.setTitleFont(border.getTitleFont().deriveFont(Font.BOLD));
        this.setBorder(border);

        this.setOpaque(false);

        jlImage = new JLabel();
        jlImage.setBorder(BorderFactory.createBevelBorder(1));
        jlText = new JLabel("");
        jlText.setVerticalAlignment(JLabel.TOP);
        jlEigenface = new JLabel();
        jlEigenface.setBorder(BorderFactory.createBevelBorder(1));

        this.add(jlImage, BorderLayout.WEST);
        this.add(jlText, BorderLayout.CENTER);
    }

    public FaceItem(Face f) {
        init();
        setFace(f);
    }

    @Override
    public Dimension getPreferredSize() {
        return getMinimumSize();
    }
}