/**
 * Front end for the 3D Face Recognition system.
 * Provides a graphical user interface for user authentication and face recognition operations.
 *
 * @author Prasad Subrahmanya
 * @version 1.0
 * @since 1.0
 */
package src;

import java.awt.*;
import java.awt.event.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.sql.*;
import javax.swing.*;
import javax.swing.border.TitledBorder;
import com.zaxxer.hikari.HikariConfig;
import com.zaxxer.hikari.HikariDataSource;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.nio.charset.StandardCharsets;
import java.util.Base64;

public class FrontEnd extends JPanel implements ActionListener {
    private static final long serialVersionUID = 1L;
    private static final String FONT_NAME = "Calibri";
    private static final int FONT_SIZE = 16;
    private static final int TEXT_FIELD_WIDTH = 30;
    private static final int PASSWORD_FIELD_WIDTH = 20;
    private static final int WINDOW_WIDTH = 700;
    private static final int WINDOW_HEIGHT = 450;
    private static final int WINDOW_X = 350;
    private static final int WINDOW_Y = 150;
    private static final String DB_URL = "jdbc:mysql://localhost/3dface";
    private static final String DB_USER = "root";
    private static final String DB_PASSWORD = "";
    private static final int MAX_POOL_SIZE = 10;
    private static final HikariDataSource dataSource;

    static {
        HikariConfig config = new HikariConfig();
        config.setJdbcUrl(DB_URL);
        config.setUsername(DB_USER);
        config.setPassword(DB_PASSWORD);
        config.setMaximumPoolSize(MAX_POOL_SIZE);
        config.addDataSourceProperty("cachePrepStmts", "true");
        config.addDataSourceProperty("prepStmtCacheSize", "250");
        config.addDataSourceProperty("prepStmtCacheSqlLimit", "2048");
        dataSource = new HikariDataSource(config);
    }

    public static final JFrame frame = new JFrame("3D Face Recognition under Expressions, Occlusions and Pose Variations");
    
    // Login panel components
    private final JPanel firstPanel;
    private final JLabel uname;
    private final JLabel pwd;
    private final JTextField un;
    private final JPasswordField pd;
    private final JButton login;
    private final JButton cancel;
    private final JButton register;

    // Registration panel components
    private final JPanel secondPanel;
    private final JLabel uname1;
    private final JLabel eml1;
    private final JLabel ph1;
    private final JLabel pwd1;
    private final JLabel cpwd1;
    private final JTextField un1;
    private final JTextField email1;
    private final JTextField pn1;
    private final JPasswordField pd1;
    private final JPasswordField cpd1;
    private final JButton reg1;
    private final JButton cancel1;

    /**
     * Creates a new FrontEnd instance and initializes the UI components.
     */
    public FrontEnd() {
        super(new GridBagLayout());
        
        // Initialize components
        firstPanel = new JPanel();
        secondPanel = new JPanel();
        
        // Initialize login components
        uname = new JLabel("Username*:");
        pwd = new JLabel("Password*:");
        un = new JTextField(TEXT_FIELD_WIDTH);
        pd = new JPasswordField(PASSWORD_FIELD_WIDTH);
        login = new JButton("Login");
        cancel = new JButton("Cancel");
        register = new JButton("Click Here!");

        // Initialize registration components
        uname1 = new JLabel("Username*:");
        eml1 = new JLabel("Email*:");
        ph1 = new JLabel("Phone*:");
        pwd1 = new JLabel("Password*:");
        cpwd1 = new JLabel("Confirm Password*:");
        un1 = new JTextField(TEXT_FIELD_WIDTH);
        email1 = new JTextField(PASSWORD_FIELD_WIDTH);
        pn1 = new JTextField(PASSWORD_FIELD_WIDTH);
        pd1 = new JPasswordField(PASSWORD_FIELD_WIDTH);
        cpd1 = new JPasswordField(PASSWORD_FIELD_WIDTH);
        reg1 = new JButton("Register");
        cancel1 = new JButton("Cancel");

        initializeUI();
    }

    /**
     * Initializes the user interface components.
     */
    private void initializeUI() {
        setupLoginPanel();
        setupRegistrationPanel();
        add(firstPanel);
    }

    /**
     * Sets up the login panel with all its components.
     */
    private void setupLoginPanel() {
        firstPanel.setLayout(new GridBagLayout());
        firstPanel.setBorder(new TitledBorder("Login"));

        // Set fonts
        uname.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));
        pwd.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));

        // Add components to panel
        GridBagConstraints gbc = createLoginConstraints();
        addLoginComponents(gbc);

        // Add action listeners
        login.addActionListener(this);
        cancel.addActionListener(this);
        register.addActionListener(this);
    }

    /**
     * Creates GridBagConstraints for the login panel.
     */
    private GridBagConstraints createLoginConstraints() {
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.anchor = GridBagConstraints.WEST;
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.insets = new Insets(10, 20, 0, 0);
        return gbc;
    }

    /**
     * Adds components to the login panel.
     */
    private void addLoginComponents(GridBagConstraints gbc) {
        firstPanel.add(uname, gbc);
        gbc.gridy++;
        gbc.insets = new Insets(20, 20, 0, 0);
        firstPanel.add(pwd, gbc);

        gbc.gridx++;
        gbc.gridy = 0;
        gbc.gridwidth = 2;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1;
        gbc.insets = new Insets(10, 10, 0, 20);
        firstPanel.add(un, gbc);
        gbc.gridy++;
        gbc.insets = new Insets(20, 10, 0, 20);
        firstPanel.add(pd, gbc);

        gbc.gridx = 1;
        gbc.gridy++;
        gbc.gridwidth = 1;
        gbc.weightx = 0;
        gbc.fill = GridBagConstraints.NONE;
        gbc.insets = new Insets(20, 35, 0, 0);
        firstPanel.add(login, gbc);
        gbc.gridx++;
        firstPanel.add(cancel, gbc);

        gbc.gridx = 1;
        gbc.gridy++;
        gbc.weighty = 0;
        gbc.gridwidth = 1;
        gbc.insets = new Insets(30, 0, 10, 20);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        firstPanel.add(new JLabel("Don't have an account?"), gbc);
        gbc.gridx++;
        gbc.insets = new Insets(30, 0, 10, 20);
        firstPanel.add(register, gbc);
    }

    /**
     * Sets up the registration panel with all its components.
     */
    private void setupRegistrationPanel() {
        secondPanel.setLayout(new GridBagLayout());
        secondPanel.setBorder(new TitledBorder("Register"));

        // Set fonts
        uname1.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));
        eml1.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));
        ph1.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));
        pwd1.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));
        cpwd1.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));

        // Add components to panel
        GridBagConstraints gbc = createRegistrationConstraints();
        addRegistrationComponents(gbc);

        // Add action listeners
        reg1.addActionListener(this);
        cancel1.addActionListener(this);
    }

    /**
     * Creates GridBagConstraints for the registration panel.
     */
    private GridBagConstraints createRegistrationConstraints() {
        GridBagConstraints gbc = new GridBagConstraints();
        gbc.gridx = 0;
        gbc.gridy = 0;
        gbc.insets = new Insets(10, 20, 0, 0);
        gbc.anchor = GridBagConstraints.WEST;
        return gbc;
    }

    /**
     * Adds components to the registration panel.
     */
    private void addRegistrationComponents(GridBagConstraints gbc) {
        secondPanel.add(uname1, gbc);
        gbc.gridy++;
        gbc.insets = new Insets(20, 20, 0, 0);
        secondPanel.add(eml1, gbc);
        gbc.gridy++;
        secondPanel.add(ph1, gbc);
        gbc.gridy++;
        secondPanel.add(pwd1, gbc);
        gbc.gridy++;
        secondPanel.add(cpwd1, gbc);

        gbc.gridx++;
        gbc.gridy = 0;
        gbc.gridwidth = 2;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1;
        gbc.insets = new Insets(10, 10, 0, 20);
        secondPanel.add(un1, gbc);
        gbc.gridy++;
        gbc.insets = new Insets(20, 10, 0, 20);
        secondPanel.add(email1, gbc);
        gbc.gridy++;
        secondPanel.add(pn1, gbc);
        gbc.gridy++;
        secondPanel.add(pd1, gbc);
        gbc.gridy++;
        secondPanel.add(cpd1, gbc);

        gbc.gridx = 1;
        gbc.gridy = 6;
        gbc.weightx = 0;
        gbc.gridwidth = 1;
        gbc.insets = new Insets(20, 35, 10, 20);
        gbc.fill = GridBagConstraints.NONE;
        secondPanel.add(reg1, gbc);
        gbc.gridx++;
        secondPanel.add(cancel1, gbc);
    }

    @Override
    public void actionPerformed(ActionEvent e) {
        Object source = e.getSource();
        if (source == login) {
            handleLogin();
        } else if (source == cancel) {
            System.exit(0);
        } else if (source == register) {
            switchToRegistration();
        } else if (source == reg1) {
            handleRegistration();
        } else if (source == cancel1) {
            switchToLogin();
        }
    }

    /**
     * Handles the login process.
     */
    private void handleLogin() {
        if (!validateLoginInput()) {
            return;
        }

        String username = un.getText();
        String password = new String(pd.getPassword());
        
        try {
            if (tryDatabaseLogin(username, password)) {
                return;
            }
            tryFileLogin(username, password);
        } catch (Exception ex) {
            JOptionPane.showMessageDialog(frame, "Login failed!");
        }
    }

    /**
     * Validates login input fields.
     */
    private boolean validateLoginInput() {
        if (un.getText().isEmpty()) {
            JOptionPane.showMessageDialog(frame, "Please enter Username!");
            return false;
        }
        if (pd.getPassword().length == 0) {
            JOptionPane.showMessageDialog(frame, "Please enter Password!");
            return false;
        }
        return true;
    }

    /**
     * Attempts to login using database credentials.
     */
    private boolean tryDatabaseLogin(String username, String password) throws SQLException {
        String hashedPassword = hashPassword(password);
        try (Connection con = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD)) {
            String query = "SELECT UserName, Password FROM users WHERE UserName = ? AND Password = ?";
            try (PreparedStatement pst = con.prepareStatement(query)) {
                pst.setString(1, username);
                pst.setString(2, hashedPassword);
                try (ResultSet res = pst.executeQuery()) {
                    if (res.next()) {
                        launchMainApplication();
                        return true;
                    }
                }
            }
        }
        return false;
    }

    /**
     * Attempts to login using file-based credentials.
     */
    private void tryFileLogin(String username, String password) {
        try (BufferedReader reader = new BufferedReader(new FileReader("users.txt"))) {
            String line = reader.readLine();
            if (line != null) {
                String[] parts = line.split("\\s");
                if (username.equals(parts[0]) && password.equals(parts[1])) {
                    launchMainApplication();
                }
            }
        } catch (Exception e) {
            // Ignore file login errors
        }
    }

    /**
     * Launches the main application window.
     */
    private void launchMainApplication() {
        JFrame j2 = new JFrame("3D Face Recognition under Expressions, Occlusions and Pose Variations");
        JApplet applet = new Main();
        j2.add(applet);
        j2.setLocation(0, 0);
        j2.setSize(1366, 730);
        j2.setVisible(true);
        applet.init();
        frame.dispose();

        j2.addWindowListener(new WindowAdapter() {
            @Override
            public void windowClosing(WindowEvent windowEvent) {
                int reply = JOptionPane.showConfirmDialog(frame,
                    "Are you sure you want to exit?", "Confirmation",
                    JOptionPane.YES_NO_OPTION);
                if (reply == JOptionPane.YES_OPTION) {
                    System.exit(0);
                }
            }
        });
        j2.setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
        j2.setVisible(true);
    }

    /**
     * Switches to the registration panel.
     */
    private void switchToRegistration() {
        remove(firstPanel);
        add(secondPanel);
        repaint();
        revalidate();
    }

    /**
     * Switches to the login panel.
     */
    private void switchToLogin() {
        remove(secondPanel);
        add(firstPanel);
        repaint();
        revalidate();
    }

    /**
     * Handles the registration process.
     */
    private void handleRegistration() {
        if (!validateRegistrationInput()) {
            return;
        }

        String username = un1.getText();
        String email = email1.getText();
        String phone = pn1.getText();
        String password = new String(pd1.getPassword());

        try {
            if (tryDatabaseRegistration(username, email, phone, password)) {
                return;
            }
            tryFileRegistration(username, password);
        } catch (Exception ex) {
            handleRegistrationError(ex);
        }
    }

    /**
     * Validates registration input fields.
     */
    private boolean validateRegistrationInput() {
        if (un1.getText().isEmpty()) {
            JOptionPane.showMessageDialog(frame, "Please enter Username!");
            return false;
        }
        if (un1.getText().length() > 30) {
            JOptionPane.showMessageDialog(frame, "Maximum 30 characters allowed for Username!");
            return false;
        }
        if (email1.getText().isEmpty()) {
            JOptionPane.showMessageDialog(frame, "Please enter Email!");
            return false;
        }
        if (!isValidEmail(email1.getText())) {
            JOptionPane.showMessageDialog(frame, "Invalid Email!");
            return false;
        }
        if (pn1.getText().isEmpty()) {
            JOptionPane.showMessageDialog(frame, "Please enter Phone!");
            return false;
        }
        if (!isValidPhone(pn1.getText())) {
            JOptionPane.showMessageDialog(frame, "Invalid Phone!");
            return false;
        }
        if (pd1.getPassword().length == 0) {
            JOptionPane.showMessageDialog(frame, "Please enter Password!");
            return false;
        }
        if (pd1.getPassword().length < 3 || pd1.getPassword().length > 15) {
            JOptionPane.showMessageDialog(frame, 
                "Password should be minimum 3 characters and maximum 15 characters!");
            return false;
        }
        if (cpd1.getPassword().length == 0) {
            JOptionPane.showMessageDialog(frame, "Please confirm password!");
            return false;
        }
        if (!new String(pd1.getPassword()).equals(new String(cpd1.getPassword()))) {
            JOptionPane.showMessageDialog(frame, "Password Mismatch: Check again!");
            return false;
        }
        return true;
    }

    /**
     * Validates email format.
     */
    private boolean isValidEmail(String email) {
        String emailPattern = "^[_A-Za-z0-9-]+(\\.[_A-Za-z0-9-]+)*@[A-Za-z0-9-]+" +
            "(\\.[A-Za-z0-9-]+)*(\\.[A-Za-z]{2,})$";
        return email.matches(emailPattern);
    }

    /**
     * Validates phone number format.
     */
    private boolean isValidPhone(String phone) {
        try {
            long p = 9999999999L;
            long r = 1000000000L;
            long q = Long.parseLong(phone, 10);
            return q <= p && q >= r;
        } catch (NumberFormatException e) {
            return false;
        }
    }

    /**
     * Attempts to register using database.
     */
    private boolean tryDatabaseRegistration(String username, String email, 
                                          String phone, String password) throws SQLException {
        String hashedPassword = hashPassword(password);
        try (Connection con = DriverManager.getConnection(DB_URL, DB_USER, DB_PASSWORD)) {
            // Check if email already exists
            String query = "SELECT Email FROM users WHERE Email = ?";
            try (PreparedStatement pst = con.prepareStatement(query)) {
                pst.setString(1, email);
                try (ResultSet res = pst.executeQuery()) {
                    if (res.next()) {
                        JOptionPane.showMessageDialog(frame, "Email already registered!");
                        return true;
                    }
                }
            }

            // Insert new user
            String insertQuery = "INSERT INTO users (UserName, Email, Phone, Password) VALUES (?, ?, ?, ?)";
            try (PreparedStatement pst = con.prepareStatement(insertQuery)) {
                pst.setString(1, username);
                pst.setString(2, email);
                pst.setString(3, phone);
                pst.setString(4, hashedPassword);
                pst.executeUpdate();
            }

            JOptionPane.showMessageDialog(frame, "Registration successful!");
            switchToLogin();
            return true;
        }
    }

    /**
     * Attempts to register using file-based storage.
     */
    private void tryFileRegistration(String username, String password) {
        try (PrintWriter writer = new PrintWriter("users.txt", "UTF-8")) {
            writer.println(username + " " + password);
            JOptionPane.showMessageDialog(frame, "Registration successful!");
            switchToLogin();
        } catch (Exception e) {
            JOptionPane.showMessageDialog(frame, "Registration failed!");
        }
    }

    /**
     * Handles registration errors.
     */
    private void handleRegistrationError(Exception ex) {
        JOptionPane.showMessageDialog(frame, "Registration failed!");
    }

    /**
     * Hashes a password using SHA-256.
     */
    private String hashPassword(String password) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(password.getBytes(StandardCharsets.UTF_8));
            return Base64.getEncoder().encodeToString(hash);
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException("Error hashing password", e);
        }
    }

    /**
     * Creates and shows the GUI.
     */
    private static void createAndShowGUI() {
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        JComponent newContentPane = new FrontEnd();
        newContentPane.setOpaque(true);
        frame.setContentPane(newContentPane);
        frame.setPreferredSize(new Dimension(WINDOW_WIDTH, WINDOW_HEIGHT));
        frame.setLocation(WINDOW_X, WINDOW_Y);
        frame.setResizable(false);
        frame.pack();
        frame.setVisible(true);
    }

    /**
     * Main method to start the application.
     */
    public static void main(String[] args) {
        SwingUtilities.invokeLater(() -> {
            try {
                UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
            } catch (Exception ex) {
                ex.printStackTrace();
            }
            createAndShowGUI();
        });
    }

    /**
     * Closes the database connection pool.
     */
    public static void closeDataSource() {
        if (dataSource != null && !dataSource.isClosed()) {
            dataSource.close();
        }
    }
}
