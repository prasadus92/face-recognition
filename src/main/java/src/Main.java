/**
 * Main application class for the face recognition system.
 * Provides the GUI interface and coordinates face recognition operations.
 *
 * @author Prasad Subrahmanya
 * @version 1.0
 * @since 1.0
 */

package src;

import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.Iterator;

import javax.imageio.ImageIO;
import javax.imageio.ImageWriter;
import javax.imageio.stream.ImageOutputStream;
import javax.swing.BorderFactory;
import javax.swing.ImageIcon;
import javax.swing.JApplet;
import javax.swing.JButton;
import javax.swing.JComponent;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JList;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JProgressBar;
import javax.swing.JScrollPane;
import javax.swing.ProgressMonitor;
import javax.swing.SwingWorker;
import javax.swing.Timer;
import javax.swing.UIManager;

public class Main extends JApplet implements ActionListener {
    private static final long serialVersionUID = 1L;
    private static final Dimension IDEAL_IMAGE_SIZE = new Dimension(48, 64);
    private static final int NUM_EIGEN_VECTORS = 10;
    private static final int CLASSIFICATION_THRESHOLD = 5;
    private static final int FACE_BROWSER_WIDTH = 400;
    private static final int FACE_BROWSER_HEIGHT = 300;
    private static final int PROGRESS_TIMER_DELAY = 100;
    private static final int WINDOW_WIDTH = 800;
    private static final int WINDOW_HEIGHT = 480;
    private static final int BUTTON_FONT_SIZE = 18;
    private static final String BUTTON_FONT = "Verdana";
    private static final int BUTTON_PADDING_Y = 30;
    private static final int BUTTON_PADDING_X = 110;
    private static final int BUTTON_INSETS = 10;

    private TSCD eigenFaces;
    private FeatureSpace featureSpace;
    private JPanel main;
    private ImageBackgroundPanel background;
    private JProgressBar statusBar;
    private JList<File> fileList;
    private JButton loadImageButton;
    private JButton trainButton;
    private JButton probeButton;
    private JButton cropImageButton;
    private ImageIcon averageFaceIcon;
    private JLabel averageFaceLabel;
    private Container container;
    private FaceItem faceCandidate;
    private FaceBrowser faceBrowser;
    private JScrollPane faceBrowserScrollPane;
    private JButton displayFeatureSpaceButton;
    private FeatureVector lastFeatureVector;
    private ArrayList<Face> faces;
    private ArrayList<FeatureVector> trainingSet;
    private String classification;

    /**
     * Creates a new Main instance.
     */
    public Main() {
        eigenFaces = new TSCD();
        featureSpace = new FeatureSpace();
        faceBrowser = new FaceBrowser();
        trainingSet = new ArrayList<>();
    }

    /**
     * Initializes the applet.
     */
    @Override
    public void init() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception exception) {
            // Use default look and feel if system look and feel is not available
        }

        container = getContentPane();
        generalInit(container);
        setSize(WINDOW_WIDTH, WINDOW_HEIGHT);
    }

    private void generalInit(Container container) {
        container.setLayout(new BorderLayout());
        main = new JPanel();
        background = new ImageBackgroundPanel();
        container.add(background, BorderLayout.CENTER);

        initializeButtons();
        initializeFaceCandidate();
        initializeStatusBar();
        initializeFaceBrowser();
        initializeRightPanel();
    }

    private void initializeButtons() {
        loadImageButton = createButton("Load Images", true);
        cropImageButton = createButton("Crop Images", false);
        trainButton = createButton("Compute Eigen Vectors", false);
        probeButton = createButton("Identify Face", false);
        displayFeatureSpaceButton = createButton("Display Result Chart", false);
    }

    private JButton createButton(String text, boolean enabled) {
        JButton button = new JButton(text);
        button.setFont(new Font(BUTTON_FONT, Font.PLAIN, BUTTON_FONT_SIZE));
        button.setEnabled(enabled);
        button.addActionListener(this);
        return button;
    }

    private void initializeFaceCandidate() {
        faceCandidate = new FaceItem();
        faceCandidate.setBorder(BorderFactory.createRaisedBevelBorder());
    }

    private void initializeStatusBar() {
        statusBar = new JProgressBar(JProgressBar.HORIZONTAL, 0, 100);
        statusBar.setBorder(BorderFactory.createEtchedBorder());
        statusBar.setStringPainted(true);
    }

    private void initializeFaceBrowser() {
        faceBrowserScrollPane = new JScrollPane(faceBrowser);
        faceBrowserScrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
        faceBrowserScrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_ALWAYS);
        faceBrowserScrollPane.setPreferredSize(new Dimension(FACE_BROWSER_WIDTH, FACE_BROWSER_HEIGHT));
        faceBrowserScrollPane.setMinimumSize(new Dimension(FACE_BROWSER_WIDTH, FACE_BROWSER_HEIGHT));
        faceBrowserScrollPane.setMaximumSize(new Dimension(FACE_BROWSER_WIDTH, FACE_BROWSER_HEIGHT));
        faceBrowserScrollPane.setSize(new Dimension(FACE_BROWSER_WIDTH, FACE_BROWSER_HEIGHT));
        faceBrowserScrollPane.setBounds(0, 0, FACE_BROWSER_WIDTH, FACE_BROWSER_HEIGHT);
    }

    private void initializeRightPanel() {
        JPanel right = new JPanel(new GridBagLayout());
        GridBagConstraints gbc = createGridBagConstraints();

        try {
            addLogoToPanel(right);
        } catch (IOException ex) {
            System.out.println("Image face.png missing\n" + ex);
        }

        addButtonsToPanel(right, gbc);
        container.add(right, BorderLayout.EAST);
    }

    private GridBagConstraints createGridBagConstraints() {
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.PAGE_START;
        gbc.gridy = 1;
        gbc.gridwidth = 4;
        gbc.ipady = BUTTON_PADDING_Y;
        gbc.ipadx = BUTTON_PADDING_X;
        gbc.insets = new Insets(BUTTON_INSETS, BUTTON_INSETS * 2, BUTTON_INSETS, BUTTON_INSETS * 2);
        return gbc;
    }

    private void addLogoToPanel(JPanel panel) throws IOException {
        String imPath = System.getProperty("user.dir").replace('\\', '/');
        BufferedImage myPicture = ImageIO.read(new File(imPath + "/src/src/face.png"));
        JLabel picLabel = new JLabel(new ImageIcon(myPicture));
        panel.add(picLabel);
    }

    private void addButtonsToPanel(JPanel panel, GridBagConstraints gbc) {
        panel.add(loadImageButton, gbc);
        gbc.gridy = 4;
        panel.add(trainButton, gbc);
        gbc.gridy = 6;
        panel.add(probeButton, gbc);
        gbc.gridy = 8;
        panel.add(displayFeatureSpaceButton, gbc);
    }

    /**
     * Handles button click events.
     */
    @Override
    public void actionPerformed(ActionEvent event) {
        Object source = event.getSource();
        if (source == loadImageButton) {
            loadImage();
        } else if (source == cropImageButton) {
            crop();
        } else if (source == trainButton) {
            train();
        } else if (source == probeButton) {
            probe();
        } else if (source == displayFeatureSpaceButton) {
            displayFeatureSpace();
        }
    }

    private void displayFeatureSpace() {
        double[][] features = featureSpace.get3dFeatureSpace(lastFeatureVector);
        if (features == null) {
            JOptionPane.showMessageDialog(this, "No feature space data available.");
            return;
        }

        JFrame frame = new JFrame("3D Face Recognition Results");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.setSize(WINDOW_WIDTH, WINDOW_HEIGHT);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private void probe() {
        try {
            JFileChooser fc = new JFileChooser();
            fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
            if (fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                long startTime = System.currentTimeMillis();
                updateStatus("Loading Files");
                File file = fc.getSelectedFile();
                try {
                    Face f = new Face(file);
                    f.load(true);
                    processFaceRecognition(f);
                    displayResults(startTime);
                } catch (MalformedURLException e) {
                    System.err.println("There was a problem opening the file : " + e.getMessage());
                }
            }
        } catch (Exception e) {
            System.err.println("There was a problem with the file chooser : " + e.getMessage());
        }
    }

    private void processFaceRecognition(Face f) {
        double[] rslt = eigenFaces.getEigenFaces(f.getPicture(), NUM_EIGEN_VECTORS);
        lastFeatureVector = new FeatureVector();
        lastFeatureVector.setFeatureVector(rslt);
        classification = featureSpace.knn(FeatureSpace.EUCLIDEAN_DISTANCE, lastFeatureVector, CLASSIFICATION_THRESHOLD);
        faceCandidate.setFace(f);
        faceCandidate.setVisible(true);
    }

    private void displayResults(long startTime) {
        long elapsedTime = System.currentTimeMillis() - startTime;
        JOptionPane.showMessageDialog(FrontEnd.frame, 
            String.format("Time Complexity of match: %.2f seconds.\nFace matched to %s.", 
                elapsedTime / 1000.0, classification));
    }

    private void loadImage() {
        try {
            faces = new ArrayList<>();
            JFileChooser fc = new JFileChooser();
            fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            
            if (fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                setupMainPanel();
                File folder = fc.getSelectedFile();
                try {
                    loadImagesFromFolder(folder);
                    setupFaceBrowser();
                    enableTrainingButtons();
                } catch (MalformedURLException e) {
                    System.err.println("There was a problem opening a file : " + e.getMessage());
                }
            }
        } catch (Exception e) {
            System.err.println("There was a problem with the file chooser : " + e.getMessage());
        }
    }

    private void setupMainPanel() {
        container.remove(background);
        container.add(main, BorderLayout.CENTER);
    }

    private void loadImagesFromFolder(File folder) throws MalformedURLException {
        File[] folders = folder.listFiles(pathname -> pathname.isDirectory());
        trainingSet.clear();
        faceBrowser.empty();

        File[] files = folder.listFiles(pathname -> 
            pathname.isFile() && (pathname.getName().endsWith(".jpg") || pathname.getName().endsWith(".png")));

        fileList.setListData(files);
        for (File file : files) {
            Face f = new Face(file);
            f.setDescription("Face image in database.");
            f.load(true);
            faces.add(f);
        }

        updateStatus(files.length + " files loaded from " + folders.length + " folders.");
    }

    private void setupFaceBrowser() {
        faceBrowserScrollPane.setViewportView(faceBrowser);
        faceBrowserScrollPane.setVisible(true);
        main.add(faceBrowserScrollPane, BorderLayout.CENTER);
    }

    private void enableTrainingButtons() {
        trainButton.setEnabled(true);
        cropImageButton.setEnabled(true);
    }

    private void crop() {
        int count = 0;
        for (Face f : faces) {
            updateProgress(count, faces.size());
            try {
                f.load(true);
            } catch (MalformedURLException e) {
                e.printStackTrace();
            }
            count++;
        }
        resetProgress();
        faceBrowser.refresh();
    }

    private void updateProgress(int count, int total) {
        int val = (count * 100) / total;
        statusBar.setValue(val);
        statusBar.setString(val + "%");
        statusBar.paintImmediately(statusBar.getVisibleRect());
    }

    private void resetProgress() {
        statusBar.setValue(0);
    }

    private void train() {
        final ProgressTracker progress = new ProgressTracker();
        Runnable calc = () -> {
            eigenFaces.processTrainingSet(faces.toArray(new Face[0]), progress);
            for (Face f : faces) {
                double[] rslt = eigenFaces.getEigenFaces(f.getPicture(), NUM_EIGEN_VECTORS);
                FeatureVector fv = new FeatureVector();
                fv.setFeatureVector(rslt);
                trainingSet.add(fv);
            }

            averageFaceIcon = new ImageIcon(getAverageFaceImage());
            averageFaceLabel.setVisible(true);
        };

        progress.run(main, calc, "Training");
    }

    private void updateStatus(String message) {
        statusBar.setString(message);
        statusBar.paintImmediately(statusBar.getVisibleRect());
    }

    public void saveImage(File f, BufferedImage img) throws IOException {
        Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName("jpg");
        ImageWriter writer = writers.next();
        try (ImageOutputStream ios = ImageIO.createImageOutputStream(f)) {
            writer.setOutput(ios);
            writer.write(img);
        }
    }

    public BufferedImage getAverageFaceImage() {
        return CreateImageFromMatrix(eigenFaces.getAverageFace().getRowPackedCopy(), IDEAL_IMAGE_SIZE.width);
    }

    public static BufferedImage CreateImageFromMatrix(double[] img, int width) {
        BufferedImage bi = new BufferedImage(width, img.length / width, BufferedImage.TYPE_BYTE_GRAY);
        for (int i = 0; i < img.length; i++) {
            int x = i % width;
            int y = i / width;
            int gray = (int) img[i];
            gray = Math.max(0, Math.min(255, gray));
            bi.setRGB(x, y, (gray << 16) | (gray << 8) | gray);
        }
        return bi;
    }

    /**
     * Helper class for tracking progress of long-running operations.
     */
    public static class ProgressTracker {
        private ProgressMonitor progressMonitor;
        private Timer timer;
        private String sProgress;
        private boolean bFinished;

        public void advanceProgress(final String message) {
            sProgress = message;
            progressMonitor.setProgress(1);
            progressMonitor.setNote(sProgress);
        }

        private class TimerListener implements ActionListener {
            @Override
            public void actionPerformed(ActionEvent evt) {
                progressMonitor.setProgress(1);
                progressMonitor.setNote(sProgress);
                if (progressMonitor.isCanceled() || bFinished) {
                    timer.stop();
                }
            }
        }

        public void run(JComponent parent, final Runnable calc, String title) {
            progressMonitor = new ProgressMonitor(parent, title, "", 0, 100);
            timer = new Timer(PROGRESS_TIMER_DELAY, new TimerListener());
            bFinished = false;
            sProgress = "Starting...";

            SwingWorker<Void, Void> worker = new SwingWorker<Void, Void>() {
                @Override
                protected Void doInBackground() {
                    calc.run();
                    return null;
                }

                @Override
                protected void done() {
                    bFinished = true;
                    progressMonitor.close();
                }
            };
            worker.execute();
            timer.start();
        }

        public void finished() {
            bFinished = true;
            progressMonitor.close();
        }
    }
}
