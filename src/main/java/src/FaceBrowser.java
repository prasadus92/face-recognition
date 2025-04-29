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

/**
 * A panel that displays a collection of face images with their classifications.
 * Provides functionality for highlighting and ordering faces based on classification.
 * Manages a scrollable view of face items, each containing an image and metadata.
 *
 * @author Prasad Subrahmanya
 * @version 1.0
 * @since 1.0
 */
public class FaceBrowser extends JPanel {

    private static final long serialVersionUID = 1L;
    private ArrayList<FaceItem> faceItems;
    private int totalHeight = 0;
    private HashMap<FaceItem, Face> itemToFaceMap;
    private HashMap<Face, FaceItem> faceToItemMap;

    /**
     * Creates a new FaceBrowser instance.
     * Initializes collections for managing face items and sets up the panel layout.
     */
    public FaceBrowser() {
        faceItems = new ArrayList<>();
        itemToFaceMap = new HashMap<>();
        faceToItemMap = new HashMap<>();
        this.setPreferredSize(new Dimension(200, 500));
        this.setBackground(this.getBackground().brighter());
    }

    /**
     * Refreshes all face items in the browser.
     * Updates the display of each face item to reflect any changes.
     */
    public void refresh() {
        for (FaceItem item : itemToFaceMap.keySet()) {
            item.refresh();
        }
    }

    /**
     * Adds a new face to the browser.
     * Creates a face item for the face and adds it to the display.
     *
     * @param face the face to add to the browser
     */
    public void addFace(Face face) {
        FaceItem item = new FaceItem(face);
        this.add(item);
        itemToFaceMap.put(item, face);
        faceToItemMap.put(face, item);
    }

    /**
     * Removes all faces from the browser.
     * Clears all collections and removes all face items from display.
     */
    public void empty() {
        this.removeAll();
        faceItems.clear();
        itemToFaceMap.clear();
        faceToItemMap.clear();
        doLayout();
    }

    /**
     * Gets the minimum size of the browser panel.
     * The width is fixed while the height accommodates all face items.
     *
     * @return the minimum dimension of the panel
     */
    @Override
    public Dimension getMinimumSize() {
        return new Dimension(256, totalHeight);
    }

    /**
     * Gets the preferred size of the browser panel.
     * Matches the minimum size to ensure proper display of all items.
     *
     * @return the preferred dimension of the panel
     */
    @Override
    public Dimension getPreferredSize() {
        return getMinimumSize();
    }

    /**
     * Highlights face items with a specific classification.
     * Adds a red border to matching items and resets others.
     *
     * @param classification the classification to highlight
     */
    public void highlightClassifiedAs(String classification) {
        for (FaceItem item : faceItems) {
            if (item.getFace().getClassification().equals(classification)) {
                item.setBorder(BorderFactory.createLineBorder(Color.RED, 2));
            } else {
                item.setBorder(BorderFactory.createRaisedBevelBorder());
            }
        }
    }

    /**
     * Reorders face items based on their distances.
     * Removes and re-adds items in the order specified by the distance pairs.
     *
     * @param faceDistances array of face-distance pairs determining the order
     */
    public void orderAs(FeatureSpace.FaceDistancePair[] faceDistances) {
        removeAll();
        for (FeatureSpace.FaceDistancePair pair : faceDistances) {
            add(new FaceItem(pair.getFace()));
        }
        revalidate();
        repaint();
    }

    /**
     * Lays out the components in the panel.
     * Arranges face items vertically and updates the total height.
     */
    @Override
    public void doLayout() {
        super.doLayout();

        Component[] components = this.getComponents();
        int currentY = 0;
        for (Component component : components) {
            component.setLocation(0, currentY);
            component.setSize(this.getWidth(), component.getHeight());
            currentY += component.getHeight();
        }

        totalHeight = currentY;
        this.revalidate();
    }
}

/**
 * A panel that displays a single face image with its classification and metadata.
 * Provides visual feedback about the face's classification status and similarity
 * to other faces through color coding and distance information.
 *
 * @author Prasad Subrahmanya
 * @version 1.0
 * @since 1.0
 */
class FaceItem extends JPanel {

    private static final long serialVersionUID = 1L;
    private Face face;
    private ImageIcon image;
    private JLabel imageLabel;
    private JLabel textLabel;
    private TitledBorder border;
    private ImageIcon eigenfaceImage;
    private JLabel eigenfaceLabel;
    private double distance = -1;

    /**
     * Creates a new empty FaceItem.
     * Initializes the panel without associating a face.
     */
    public FaceItem() {
        init();
    }

    /**
     * Sets the distance value for this face item.
     * Updates the display to reflect the similarity to a reference face.
     *
     * @param dist the distance value to set
     */
    public void setDistance(double dist) {
        this.distance = dist;

        updateLabel();

        double amt = dist / 4096;
        if (amt > 1) {
            amt = 1;
        }
        amt = 0.5 + amt / 2;
        this.setBackground(new Color((float) amt, 1.0f, (float) amt));
        this.setOpaque(true);


    }

    /**
     * Updates the text label with current face information.
     * Displays classification, distance, description, and file path.
     */
    private void updateLabel() {
        String text = "<html>";
        text += "<font size=+1><font color=#7f7f7f>Classification:</font> ";
        if (this.face.getClassification() == null) {
            text += "<font color=#7f0000>[unclassified]</font>";
        } else {
            text += "<font color=#00007f>" + this.face.getClassification() + "</font>";
        }
        text += "</b></font>";

        if (this.distance >= 0) {
            text += ("<br><b>" + "Distance: " + this.distance + "</b>");
        }

        text += "<br>" + this.face.getDescription() + "";
        text += "<br><font size=-2 color=#7f7f7f>" + this.face.getFile().getAbsolutePath() + "</font>";
        text += "</html>";
        textLabel.setText(text);
    }

    /**
     * Sets the highlight state of this face item.
     * Changes the border and opacity to indicate selection.
     *
     * @param b true to highlight, false to remove highlight
     */
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

    /**
     * Refreshes the face image display.
     * Updates the image icon with the current face picture.
     */
    public void refresh() {
        this.image = new ImageIcon(this.face.getPicture().getImage());
        imageLabel.setIcon(this.image);
    }

    /**
     * Associates a face with this item and updates the display.
     * Sets the face image, title, and metadata information.
     *
     * @param f the face to associate with this item
     */
    public void setFace(Face f) {
        this.face = f;
        refresh();
        border.setTitle(f.getFile().getName());
        updateLabel();
        Insets i = imageLabel.getInsets();
        imageLabel.setPreferredSize(
                new Dimension(
                image.getIconWidth() + i.left + i.right,
                image.getIconHeight() + i.top + i.bottom));
    }

    /**
     * Initializes the face item components.
     * Sets up the layout and creates necessary UI elements.
     */
    private void init() {
        BorderLayout layout = new BorderLayout();
        this.setLayout(layout);

        border = BorderFactory.createTitledBorder(BorderFactory.createLineBorder(this.getBackground()), "");
        //border.setTitleFont(border.getTitleFont().deriveFont(Font.BOLD));
        this.setBorder(border);

        this.setOpaque(false);

        imageLabel = new JLabel();
        imageLabel.setBorder(BorderFactory.createBevelBorder(1));
        textLabel = new JLabel("");
        textLabel.setVerticalAlignment(JLabel.TOP);
        eigenfaceLabel = new JLabel();
        eigenfaceLabel.setBorder(BorderFactory.createBevelBorder(1));

        this.add(imageLabel, BorderLayout.WEST);
        this.add(textLabel, BorderLayout.CENTER);
    }

    public FaceItem(Face f) {
        init();
        setFace(f);
    }

    @Override
    public Dimension getPreferredSize() {
        return getMinimumSize();
    }

    public Face getFace() {
        return face;
    }
}