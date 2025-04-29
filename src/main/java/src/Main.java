/**
 * Main application class for the face recognition system.
 * Provides the GUI interface and coordinates face recognition operations.
 * This class serves as the entry point and controller for the face recognition
 * application, managing the user interface, image processing, and recognition workflow.
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
     * Initializes the core components including eigenfaces processor,
     * feature space, face browser, and training set storage.
     */
    public Main() {
        eigenFaces = new TSCD();
        featureSpace = new FeatureSpace();
        faceBrowser = new FaceBrowser();
        trainingSet = new ArrayList<>();
    }

    /**
     * Initializes the applet with system look and feel.
     * Sets up the main container and initializes all UI components.
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

    /**
     * Initializes the general layout and components of the application.
     * Sets up the main panel, background, buttons, face candidate,
     * status bar, face browser, and right panel.
     *
     * @param container the main container to initialize
     */
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

    /**
     * Initializes all control buttons with appropriate text and state.
     * Creates buttons for loading images, cropping, computing eigenvectors,
     * face identification, and result display.
     */
    private void initializeButtons() {
        loadImageButton = createButton("Load Images", true);
        cropImageButton = createButton("Crop Images", false);
        trainButton = createButton("Compute Eigen Vectors", false);
        probeButton = createButton("Identify Face", false);
        displayFeatureSpaceButton = createButton("Display Result Chart", false);
    }

    /**
     * Creates a styled button with the specified text and enabled state.
     *
     * @param text the button text
     * @param enabled whether the button should be initially enabled
     * @return the created and configured button
     */
    private JButton createButton(String text, boolean enabled) {
        JButton button = new JButton(text);
        button.setFont(new Font(BUTTON_FONT, Font.PLAIN, BUTTON_FONT_SIZE));
        button.setEnabled(enabled);
        button.addActionListener(this);
        return button;
    }

    /**
     * Initializes the face candidate component for displaying
     * the current face being processed.
     */
    private void initializeFaceCandidate() {
        faceCandidate = new FaceItem();
        faceCandidate.setBorder(BorderFactory.createRaisedBevelBorder());
    }

    /**
     * Initializes the status bar for showing progress information.
     * Creates a horizontal progress bar with percentage display.
     */
    private void initializeStatusBar() {
        statusBar = new JProgressBar(JProgressBar.HORIZONTAL, 0, 100);
        statusBar.setBorder(BorderFactory.createEtchedBorder());
        statusBar.setStringPainted(true);
    }

    /**
     * Initializes the face browser component for displaying
     * multiple face images in a scrollable view.
     */
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

    /**
     * Initializes the right panel containing the logo and control buttons.
     * Sets up the layout and adds all necessary components.
     */
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

    /**
     * Creates and configures GridBagConstraints for button layout.
     *
     * @return configured GridBagConstraints object
     */
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

    /**
     * Adds the application logo to the specified panel.
     *
     * @param panel the panel to add the logo to
     * @throws IOException if the logo image file cannot be read
     */
    private void addLogoToPanel(JPanel panel) throws IOException {
        String imPath = System.getProperty("user.dir").replace('\\', '/');
        BufferedImage myPicture = ImageIO.read(new File(imPath + "/src/src/face.png"));
        JLabel picLabel = new JLabel(new ImageIcon(myPicture));
        panel.add(picLabel);
    }

    /**
     * Adds all control buttons to the specified panel using the given constraints.
     *
     * @param panel the panel to add buttons to
     * @param gbc the GridBagConstraints to use for layout
     */
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
     * Handles button click events from the UI.
     * Routes each action to its appropriate handler method.
     *
     * @param event the action event containing the source button
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

    /**
     * Displays the feature space visualization.
     * Creates and shows a 3D chart of the feature space with the current
     * feature vector highlighted.
     */
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

    /**
     * Processes a probe image for face recognition.
     * Loads the selected image, extracts features, and attempts to
     * identify the face using the trained eigenfaces.
     */
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

    /**
     * Processes face recognition for a given face.
     * Extracts features and performs k-nearest neighbor classification.
     *
     * @param f the face to process and classify
     */
    private void processFaceRecognition(Face f) {
        double[] rslt = eigenFaces.getEigenFaces(f.getPicture(), NUM_EIGEN_VECTORS);
        lastFeatureVector = new FeatureVector();
        lastFeatureVector.setFeatureVector(rslt);
        classification = featureSpace.knn(FeatureSpace.EUCLIDEAN_DISTANCE, lastFeatureVector, CLASSIFICATION_THRESHOLD);
        faceCandidate.setFace(f);
        faceCandidate.setVisible(true);
    }

    /**
     * Displays the recognition results and processing time.
     *
     * @param startTime the time when processing started
     */
    private void displayResults(long startTime) {
        long elapsedTime = System.currentTimeMillis() - startTime;
        JOptionPane.showMessageDialog(FrontEnd.frame, 
            String.format("Time Complexity of match: %.2f seconds.\nFace matched to %s.", 
                elapsedTime / 1000.0, classification));
    }

    /**
     * Handles the load image button action.
     * Opens a file chooser for selecting face images and loads them
     * into the face browser.
     */
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

    /**
     * Sets up the main panel for displaying face images.
     * Removes the background and adds the main content panel.
     */
    private void setupMainPanel() {
        container.remove(background);
        container.add(main, BorderLayout.CENTER);
    }

    /**
     * Loads all face images from a specified folder.
     * Processes only .jpg and .png files, creating Face objects
     * for each valid image.
     *
     * @param folder the folder containing face images
     * @throws MalformedURLException if an image file URL is malformed
     */
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

    /**
     * Sets up the face browser component in the main panel.
     * Configures the scroll pane and makes it visible.
     */
    private void setupFaceBrowser() {
        faceBrowserScrollPane.setViewportView(faceBrowser);
        faceBrowserScrollPane.setVisible(true);
        main.add(faceBrowserScrollPane, BorderLayout.CENTER);
    }

    /**
     * Enables the training-related buttons after images are loaded.
     */
    private void enableTrainingButtons() {
        trainButton.setEnabled(true);
        cropImageButton.setEnabled(true);
    }

    /**
     * Initiates the face cropping process for loaded images.
     */
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

    /**
     * Updates the progress bar with current progress.
     *
     * @param count current progress count
     * @param total total number of steps
     */
    private void updateProgress(int count, int total) {
        int val = (count * 100) / total;
        statusBar.setValue(val);
        statusBar.setString(val + "%");
        statusBar.paintImmediately(statusBar.getVisibleRect());
    }

    /**
     * Resets the progress bar to zero.
     */
    private void resetProgress() {
        statusBar.setValue(0);
    }

    /**
     * Initiates the training process using loaded face images.
     * Computes eigenfaces and builds the feature space.
     */
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

    /**
     * Updates the status message in the progress bar.
     *
     * @param message the status message to display
     */
    private void updateStatus(String message) {
        statusBar.setString(message);
        statusBar.paintImmediately(statusBar.getVisibleRect());
    }

    /**
     * Saves a BufferedImage to a file.
     *
     * @param f the file to save to
     * @param img the image to save
     * @throws IOException if there is an error writing the file
     */
    public void saveImage(File f, BufferedImage img) throws IOException {
        Iterator<ImageWriter> writers = ImageIO.getImageWritersByFormatName("jpg");
        ImageWriter writer = writers.next();
        try (ImageOutputStream ios = ImageIO.createImageOutputStream(f)) {
            writer.setOutput(ios);
            writer.write(img);
        }
    }

    /**
     * Gets the average face image from the eigenfaces computation.
     *
     * @return the average face as a BufferedImage
     */
    public BufferedImage getAverageFaceImage() {
        return CreateImageFromMatrix(eigenFaces.getAverageFace().getRowPackedCopy(), IDEAL_IMAGE_SIZE.width);
    }

    /**
     * Creates a BufferedImage from a matrix of pixel values.
     *
     * @param img the array of pixel values
     * @param width the width of the image
     * @return the created BufferedImage
     */
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
     * Inner class for tracking progress of long-running operations.
     * Provides methods for updating and displaying progress information.
     */
    public static class ProgressTracker {
        private ProgressMonitor progressMonitor;
        private Timer timer;
        private String sProgress;
        private boolean bFinished;

        /**
         * Updates the progress message.
         *
         * @param message the new progress message
         */
        public void advanceProgress(final String message) {
            sProgress = message;
            progressMonitor.setProgress(1);
            progressMonitor.setNote(sProgress);
        }

        /**
         * Timer listener for updating the progress monitor.
         */
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

        /**
         * Runs a task with progress monitoring.
         *
         * @param parent the parent component for the progress dialog
         * @param calc the task to run
         * @param title the title for the progress dialog
         */
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

        /**
         * Marks the progress tracking as finished.
         */
        public void finished() {
            bFinished = true;
            progressMonitor.close();
        }
    }
}
