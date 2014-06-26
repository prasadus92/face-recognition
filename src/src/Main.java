/* Author: Prasad U S
*
*
*/

package src;

import java.lang.Exception;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.BorderLayout;
import java.awt.Canvas;
import java.awt.Color;
import java.awt.ComponentOrientation;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.GridLayout;
import java.awt.Insets;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileFilter;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.Arrays;
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
import javax.swing.JTextField;
import javax.swing.ProgressMonitor;
import javax.swing.Timer;
import javax.swing.UIManager;

import uk.co.chartbuilder.data.DataSet;
import uk.co.chartbuilder.examples.facialrecognition.ResultDataParser;
import uk.co.chartbuilder.examples.facialrecognition.ResultsChart;
import uk.co.chartbuilder.parser.ParserException;

public class Main extends JApplet implements ActionListener {

    public static String classification;
    public static final Dimension IDEAL_IMAGE_SIZE = new Dimension(48, 64);
    TSCD eigenFaces = new TSCD();
    FeatureSpace featureSpace = new FeatureSpace();
    private static final long serialVersionUID = 1L;
    JPanel main;
    ImageBackgroundPanel bkd;
    // JLabel jlClassthreshold;
    JProgressBar jlStatus;
    JList jlist;
    JButton jbLoadImage, jbTrain, jbProbe, jbCropImage;
    ImageIcon imageAverageFace;
    JLabel jlAverageFace;
    Container c;
    FaceItem faceCandidate;
    FaceBrowser faceBrowser = new FaceBrowser();
    private JScrollPane jspFaceBrowser;
    JButton jbDisplayFeatureSpace;
    //JTextField jtfClassthreshold;
    int classthreshold = 5;
    FeatureVector lastFV = null;
    ArrayList<Face> faces;
    DataSet resultsData;

    //public static void main(String[] args) {
    //Main m = new Main();
    //System.out.println("OOPS");
    //}
    public void generalInit(Container c) {
        c.setLayout(new BorderLayout());
        main = new JPanel();

        bkd = new ImageBackgroundPanel();
        c.add(bkd, "Center");

        //c.add(main, "Center");
        //main.add(bckgd);

        jbLoadImage = new JButton("Load Images");
        jbLoadImage.addActionListener(this);
        jbCropImage = new JButton("Crop Images");
        jbCropImage.addActionListener(this);
        jbCropImage.setEnabled(false);
        jbTrain = new JButton("Compute Eigen Vectors");
        jbTrain.setEnabled(false);
        jbTrain.addActionListener(this);
        jbProbe = new JButton("Identify Face");
        jbProbe.addActionListener(this);
        jbProbe.setEnabled(false);
        jbDisplayFeatureSpace = new JButton("Display Result Chart");
        jbDisplayFeatureSpace.addActionListener(this);
        jbDisplayFeatureSpace.setEnabled(false);
        //jlClassthreshold = new JLabel("Factor");
        // jtfClassthreshold = new JTextField(""+classthreshold);
        //jbClassthreshold = new JButton("Update Threshold");
        // jbClassthreshold.addActionListener(this);

        faceCandidate = new FaceItem();
        faceCandidate.setBorder(BorderFactory.createRaisedBevelBorder());

        jlAverageFace = new JLabel();
        jlAverageFace.setVerticalTextPosition(JLabel.BOTTOM);
        jlAverageFace.setHorizontalTextPosition(JLabel.CENTER);

        jlStatus = new JProgressBar(JProgressBar.HORIZONTAL, 0, 100);
        jlStatus.setBorder(BorderFactory.createEtchedBorder());
        jlStatus.setStringPainted(true);
        jlist = new JList();
        main.setLayout(new BorderLayout());
        JPanel right = new JPanel();

        jbLoadImage.setFont(new Font("Verdana", 30, 18));
        //jbCropImage.setFont(new Font("Cambria", 20, 28));
        jbTrain.setFont(new Font("Verdana", 30, 18));
        jbProbe.setFont(new Font("Verdana", 30, 18));
        jbDisplayFeatureSpace.setFont(new Font("Verdana", 30, 18));
        right.setLayout(new GridBagLayout());
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.anchor = GridBagConstraints.PAGE_START;

        gbc.gridy = 1;
        gbc.gridwidth = 4;
        gbc.ipady = 30;
        gbc.ipadx = 110;
        gbc.insets = new Insets(10, 20, 10, 20);

        //JLabel myp = new JLabel("Project ID: PW13A01");
        //myp.setFont(new Font("Tahoma", 20, 20));
        //right.add(myp);
        //gbc.gridy = 3;

        try {
            String imPath = System.getProperty("user.dir");
            imPath = imPath.replace('\\', '/');
            BufferedImage myPicture = ImageIO.read(new File(imPath + "/src/src/face.png"));
            JLabel picLabel = new JLabel(new ImageIcon(myPicture));
            //picLabel.setSize(250, 220);
            right.add(picLabel);
        } catch (IOException ex) {
            System.out.println("Image face.png missing\n" + ex);
        }


        right.add(jbLoadImage, gbc);
        //gbc.gridy = 1; right.add(jbCropImage, gbc);
        gbc.gridy = 4;
        right.add(jbTrain, gbc);
        gbc.gridy = 6;
        right.add(jbProbe, gbc);
        gbc.gridy = 8;
        right.add(jbDisplayFeatureSpace, gbc);
        //gbc.gridy = 5; gbc.gridwidth = 1; right.add(jlClassthreshold, gbc);
        // gbc.gridy = 5; gbc.gridwidth = 1; right.add(jtfClassthreshold, gbc);
        // gbc.gridy = 6; gbc.gridwidth = 2; right.add(jbClassthreshold, gbc);
       /* gbc.gridy = 7;
         gbc.weighty = 1.0;
         gbc.fill = GridBagConstraints.VERTICAL | GridBagConstraints.HORIZONTAL;
         right.add(jlAverageFace, gbc);  */

        c.add(right, BorderLayout.EAST);


        //Mark



    }

    //public Main(){
    //try {
    //UIManager.setLookAndFeel("com.sun.java.swing.plaf.windows.WindowsLookAndFeel");
    //} catch (Exception exception) {}
    //}
    public void init() {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception exception) {
        }

        c = getContentPane();
        generalInit(c);
        setSize(800, 480);
    }

    public void actionPerformed(ActionEvent arg0) {



        if (arg0.getSource() == jbLoadImage) {
            loadImage();



        } else if (arg0.getSource() == jbCropImage) {
            crop();
        } else if (arg0.getSource() == jbTrain) {
            train();
        } else if (arg0.getSource() == jbProbe) {
            probe();
        } else if (arg0.getSource() == jbDisplayFeatureSpace) {
            displayFeatureSpace();
        }
        //else if(arg0.getSource() == jbClassthreshold) updateThreshold();
    }

    //public void updateThreshold() {
    // classthreshold = Integer.parseInt(jtfClassthreshold.getText());
    //}
    private void displayFeatureSpace() {
        double[][] features = featureSpace.get3dFeatureSpace(lastFV);
        ResultDataParser parser = new ResultDataParser(features);
        try {
            parser.parse();
        } catch (ParserException pe) {
            System.out.println(pe.toString());
            System.exit(1);
        }


        JFrame frame = new JFrame("3D Face Recognition Results Chart");
        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
        frame.getContentPane().setLayout(new BorderLayout());

        resultsData = parser;
        Canvas resultsCanvas = ResultsChart.getCanvas();
        JPanel resultsPanel = new JPanel();
        resultsPanel.setOpaque(false);
        resultsPanel.setLayout(new BorderLayout());
        resultsPanel.add(resultsCanvas, BorderLayout.CENTER);

        frame.getContentPane().add(resultsPanel, BorderLayout.CENTER);

        JLabel lbl = new JLabel("3D Face Recognition");
        lbl.setBackground(Color.BLACK);
        lbl.setForeground(Color.WHITE);
        lbl.setOpaque(true);
        lbl.setFont(lbl.getFont().deriveFont(Font.BOLD));
        frame.getContentPane().add(lbl, BorderLayout.SOUTH);
        ResultsChart resultsChart =
                new ResultsChart(resultsCanvas, resultsData);

        frame.setSize(800, 720);
        frame.setResizable(false);
        frame.setLocationRelativeTo(null);
        frame.setVisible(true);
    }

    private void probe() {
        double et = 0;
        try {
            JFileChooser fc = new JFileChooser();
            fc.setDialogTitle("Load a file");
            fc.setFileSelectionMode(JFileChooser.FILES_ONLY);
            if (fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
                //Calculate the time complexity of matching
                long startTime = System.currentTimeMillis();  //Starting time

                jlStatus.setString("Loading Files");
                jlStatus.paintImmediately(jlStatus.getVisibleRect());
                File file = fc.getSelectedFile();



                Face f = new Face(file);
                f.load(true);


                int numVecs = 10;
                double[] rslt = eigenFaces.getEigenFaces(f.picture, numVecs);
                FeatureVector fv = new FeatureVector();
                fv.setFeatureVector(rslt);

                classification = featureSpace.knn(FeatureSpace.EUCLIDEAN_DISTANCE, fv, classthreshold);
                f.classification = classification;
                f.description = "Query face image.";

                faceBrowser.highlightClassifiedAs(classification);

                FeatureSpace.fd_pair[] faceDistances = featureSpace.orderByDistance(FeatureSpace.EUCLIDEAN_DISTANCE, fv);

                //Not matching scenario
                //Distance more than 3000
                FeatureSpace.fd_pair fd = faceDistances[0];
                long st = System.currentTimeMillis();
                et = st - startTime;
                et /= 1000.0;
                if (fd.dist > 800) {
                    Exception e = new Exception();
                    throw e;
                } else {

                    if (et >= 8) {
                        Exception e1 = new Exception();
                        throw e1;
                    }
                }
                //Not matching scenario ends

                faceBrowser.orderAs(faceDistances);

                lastFV = fv;

                jlStatus.setIndeterminate(false);
                jlStatus.setString("Face matched to " + classification);
                jlStatus.paintImmediately(jlStatus.getVisibleRect());
                faceCandidate.setFace(f);
                faceCandidate.setVisible(true);


                long stopTime = System.currentTimeMillis();  //Stopping time
                long elapsedTime = stopTime - startTime;  //Calculate time elapsed in milliseconds
                JOptionPane.showMessageDialog(FrontEnd.frame, "Time Complexity of match: " + ((double) (elapsedTime / 1000.0)) + " seconds.\nFace matched to " + classification + ".");
                //Display the time as a message dialog in seconds
            }
        } catch (MalformedURLException e) {
            System.err.println("There was a problem opening a file : " + e.getMessage());
            e.printStackTrace(System.err);
        } catch (Exception e) {
            JOptionPane.showMessageDialog(this, "Error: Image not matched to any of the database images!!\nNo match found!!\nTime elapsed: " + et + " seconds.");
        }
    }

    private void loadImage() {
        try {


            faces = new ArrayList<Face>();

            JFileChooser fc = new JFileChooser();
            fc.setDialogTitle("Load a file");
            fc.setFileSelectionMode(JFileChooser.DIRECTORIES_ONLY);
            if (fc.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {

                c.remove(bkd);
                c.add(main, "Center");

                main.add(jlStatus, BorderLayout.SOUTH);
                main.add(faceCandidate, BorderLayout.NORTH);
                faceCandidate.setVisible(false);
                faceCandidate.setBackground(Color.WHITE);
                faceCandidate.setOpaque(true);
                jspFaceBrowser = new JScrollPane(faceBrowser);
                main.add(jspFaceBrowser, BorderLayout.CENTER);

                repaint();
                jlStatus.setString("Loading Files");
                jlStatus.paintImmediately(jlStatus.getVisibleRect());
                ArrayList<File> trainingSet = new ArrayList<File>();

                File folder = fc.getSelectedFile();
                //System.out.println("1	"+folder);
                FileFilter dirFilter = new FileFilter() {
                    public boolean accept(File pathname) {
                        return pathname.exists() && pathname.isDirectory();
                    }
                };
                FileFilter jpgFilter = new FileFilter() {
                    public boolean accept(File pathname) {
                        String filename = pathname.getName();
                        boolean jpgFile = (filename.toUpperCase().endsWith("JPG")
                                || filename.toUpperCase().endsWith("JPEG"));
                        return pathname.exists() && pathname.isFile() && jpgFile;
                    }
                };

                File[] folders = folder.listFiles(dirFilter);
                //System.out.println("2	"+folders);
                trainingSet.clear();
                faceBrowser.empty();

                for (int i = 0; i < folders.length; i++) {				//For each folder in the training set directory
                    File[] files = folders[i].listFiles(jpgFilter);
                    System.out.println("3	" + files);
                    for (int j = 0; j < files.length; j++) {
                        trainingSet.add(files[j]);
                    }
                }

                File[] files = trainingSet.toArray(new File[1]);

                jlist.setListData(files);
                //there is no image files in the folderwai
                //System.out.println(files);
                for (int i = 0; i < files.length; i++) {
                    //System.out.println(files[0]);
                    Face f = new Face(files[i]);
                    f.description = "Face image in database.";
                    f.classification = files[i].getParentFile().getName();
                    faceBrowser.addFace(f);
                    faces.add(f);
                }

                jlStatus.setIndeterminate(false);
                jlStatus.setString(files.length + " files loaded from " + folders.length + " folders.");
                jlStatus.paintImmediately(jlStatus.getVisibleRect());


                jspFaceBrowser.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_ALWAYS);
                main.invalidate();

                jbTrain.setEnabled(true);
                jbCropImage.setEnabled(true);
            }

        } catch (MalformedURLException e) {
            System.err.println("There was a problem opening a file : " + e.getMessage());
            e.printStackTrace(System.err);
        }
    }

    private void crop() {
        int count = 0;
        for (Face f : faces) {
            int val = (count * 100) / faces.size();
            jlStatus.setValue(val);
            jlStatus.setString(val + "%");
            jlStatus.paintImmediately(jlStatus.getVisibleRect());
            try {
                f.load(true);
            } catch (MalformedURLException e) {
                e.printStackTrace();
            }
            count++;
            jlStatus.paintImmediately(jlStatus.getVisibleRect());
        }
        jlStatus.setValue(0);
        faceBrowser.refresh();
    }

    private void train() {
        final ProgressTracker progress = new ProgressTracker();

        Runnable calc = new Runnable() {
            public void run() {
                featureSpace = new FeatureSpace();

                eigenFaces.processTrainingSet(faces.toArray(new Face[0]), progress);

                for (Face f : faces) {

                    int numVecs = 10;
                    double[] rslt = eigenFaces.getEigenFaces(f.picture, numVecs); //
                    featureSpace.insertIntoDatabase(f, rslt);
                }

                jbProbe.setEnabled(true);
                jbDisplayFeatureSpace.setEnabled(true);

                imageAverageFace = new ImageIcon(getAverageFaceImage());
                jlAverageFace.setVisible(true);
                //jlAverageFace.setIcon(imageAverageFace);
                //jlAverageFace.setText("Face Average");
            }
        };

        progress.run(main, calc, "Training");


    }

    public void saveImage(File f, BufferedImage img) throws IOException {

        Iterator writers = ImageIO.getImageWritersByFormatName("jpg");
        ImageWriter writer = (ImageWriter) writers.next();

        ImageOutputStream ios = ImageIO.createImageOutputStream(f);
        writer.setOutput(ios);

        writer.write(img);

        ios.close();
    }

    public BufferedImage getAverageFaceImage() {
        return Main.CreateImageFromMatrix(eigenFaces.averageFace.getRowPackedCopy(), IDEAL_IMAGE_SIZE.width);
    }

    public static BufferedImage CreateImageFromMatrix(double[] img, int width) {
        int[] grayImage = new int[img.length];
        double[] scales = (double[]) img.clone();
        Arrays.sort(scales);
        double min = scales[0];
        double max = scales[scales.length - 1];

        for (int i = 0; i < grayImage.length; i++) {
            double v = img[i];
            v -= min;
            v /= (max - min);
            short val = (short) (v * 255);
            grayImage[i] = (val << 16) | (val << 8) | (val);
        }
        BufferedImage bi = new BufferedImage(width, img.length / width, BufferedImage.TYPE_INT_RGB);
        bi.setRGB(0, 0, width, img.length / width, grayImage, 0, width);
        return bi;
    }

    class ProgressTracker {

        Thread thread;
        int task = 0;
        private ProgressMonitor progressMonitor;
        private Timer timer;
        private String sProgress;
        private boolean bFinished;

        public void advanceProgress(final String message) {
            task++;
            System.out.println(message);
            sProgress = "Task " + task + ": " + message;
        }

        class TimerListener implements ActionListener {

            public void actionPerformed(ActionEvent evt) {
                progressMonitor.setProgress(1);
                progressMonitor.setNote(sProgress);
                if (progressMonitor.isCanceled() || bFinished) {

                    timer.stop();
                }
            }
        }

        public void run(JComponent parent, final Runnable calc, String title) {
            bFinished = false;
            progressMonitor = new ProgressMonitor(parent,
                    title, "", 0, 100);
            progressMonitor.setProgress(0);
            progressMonitor.setMillisToDecideToPopup(0);

            timer = new Timer(100, new TimerListener());

            final SwingWorker worker = new SwingWorker() {
                public Object construct() {
                    thread = new Thread(calc);
                    thread.setPriority(Thread.MIN_PRIORITY);
                    thread.start();
                    return null;
                }
            };
            worker.start();
            timer.start();

        }

        public void finished() {
            bFinished = true;
            progressMonitor.close();
            timer.stop();
        }
    }
}
