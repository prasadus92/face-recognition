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

    public static final JFrame frame = new JFrame("3D Face Recognition under Expressions, Occlusions and Pose Variations");
    
    // Login panel components
    private final JPanel loginPanel;
    private final JLabel usernameLabel;
    private final JLabel passwordLabel;
    private final JTextField usernameField;
    private final JPasswordField passwordField;
    private final JButton loginButton;
    private final JButton cancelButton;
    private final JButton registerButton;

    // Registration panel components
    private final JPanel registrationPanel;
    private final JLabel regUsernameLabel;
    private final JLabel emailLabel;
    private final JLabel phoneLabel;
    private final JLabel regPasswordLabel;
    private final JLabel confirmPasswordLabel;
    private final JTextField regUsernameField;
    private final JTextField emailField;
    private final JTextField phoneField;
    private final JPasswordField regPasswordField;
    private final JPasswordField confirmPasswordField;
    private final JButton registerSubmitButton;
    private final JButton regCancelButton;

    /**
     * Creates a new FrontEnd instance and initializes the UI components.
     */
    public FrontEnd() {
        super(new GridBagLayout());
        
        // Initialize components
        loginPanel = new JPanel();
        registrationPanel = new JPanel();
        
        // Initialize login components
        usernameLabel = new JLabel("Username*:");
        passwordLabel = new JLabel("Password*:");
        usernameField = new JTextField(TEXT_FIELD_WIDTH);
        passwordField = new JPasswordField(PASSWORD_FIELD_WIDTH);
        loginButton = new JButton("Login");
        cancelButton = new JButton("Cancel");
        registerButton = new JButton("Click Here!");

        // Initialize registration components
        regUsernameLabel = new JLabel("Username*:");
        emailLabel = new JLabel("Email*:");
        phoneLabel = new JLabel("Phone*:");
        regPasswordLabel = new JLabel("Password*:");
        confirmPasswordLabel = new JLabel("Confirm Password*:");
        regUsernameField = new JTextField(TEXT_FIELD_WIDTH);
        emailField = new JTextField(PASSWORD_FIELD_WIDTH);
        phoneField = new JTextField(PASSWORD_FIELD_WIDTH);
        regPasswordField = new JPasswordField(PASSWORD_FIELD_WIDTH);
        confirmPasswordField = new JPasswordField(PASSWORD_FIELD_WIDTH);
        registerSubmitButton = new JButton("Register");
        regCancelButton = new JButton("Cancel");

        initializeUI();
    }

    /**
     * Initializes the user interface components.
     */
    private void initializeUI() {
        setupLoginPanel();
        setupRegistrationPanel();
        add(loginPanel);
    }

    /**
     * Sets up the login panel with all its components.
     */
    private void setupLoginPanel() {
        loginPanel.setLayout(new GridBagLayout());
        loginPanel.setBorder(new TitledBorder("Login"));

        // Set fonts
        usernameLabel.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));
        passwordLabel.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));

        // Add components to panel
        GridBagConstraints gbc = createLoginConstraints();
        addLoginComponents(gbc);

        // Add action listeners
        loginButton.addActionListener(this);
        cancelButton.addActionListener(this);
        registerButton.addActionListener(this);
    }

    /**
     * Creates GridBagConstraints for the login panel.
     * Configures constraints for proper component layout in the login form.
     *
     * @return configured GridBagConstraints for login panel layout
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
     * Adds components to the login panel using the specified constraints.
     * Arranges username and password fields, buttons, and registration link.
     *
     * @param gbc the GridBagConstraints to use for component layout
     */
    private void addLoginComponents(GridBagConstraints gbc) {
        loginPanel.add(usernameLabel, gbc);
        gbc.gridy++;
        gbc.insets = new Insets(20, 20, 0, 0);
        loginPanel.add(passwordLabel, gbc);

        gbc.gridx++;
        gbc.gridy = 0;
        gbc.gridwidth = 2;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1;
        gbc.insets = new Insets(10, 10, 0, 20);
        loginPanel.add(usernameField, gbc);
        gbc.gridy++;
        gbc.insets = new Insets(20, 10, 0, 20);
        loginPanel.add(passwordField, gbc);

        gbc.gridx = 1;
        gbc.gridy++;
        gbc.gridwidth = 1;
        gbc.weightx = 0;
        gbc.fill = GridBagConstraints.NONE;
        gbc.insets = new Insets(20, 35, 0, 0);
        loginPanel.add(loginButton, gbc);
        gbc.gridx++;
        loginPanel.add(cancelButton, gbc);

        gbc.gridx = 1;
        gbc.gridy++;
        gbc.weighty = 0;
        gbc.gridwidth = 1;
        gbc.insets = new Insets(30, 0, 10, 20);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        loginPanel.add(new JLabel("Don't have an account?"), gbc);
        gbc.gridx++;
        gbc.insets = new Insets(30, 0, 10, 20);
        loginPanel.add(registerButton, gbc);
    }

    /**
     * Sets up the registration panel with all its components.
     */
    private void setupRegistrationPanel() {
        registrationPanel.setLayout(new GridBagLayout());
        registrationPanel.setBorder(new TitledBorder("Register"));

        // Set fonts
        regUsernameLabel.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));
        emailLabel.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));
        phoneLabel.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));
        regPasswordLabel.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));
        confirmPasswordLabel.setFont(new Font(FONT_NAME, Font.BOLD, FONT_SIZE));

        // Add components to panel
        GridBagConstraints gbc = createRegistrationConstraints();
        addRegistrationComponents(gbc);

        // Add action listeners
        registerSubmitButton.addActionListener(this);
        regCancelButton.addActionListener(this);
    }

    /**
     * Creates GridBagConstraints for the registration panel.
     * Configures constraints for proper component layout in the registration form.
     *
     * @return configured GridBagConstraints for registration panel layout
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
     * Adds components to the registration panel using the specified constraints.
     * Arranges username, email, phone, password fields, and buttons.
     *
     * @param gbc the GridBagConstraints to use for component layout
     */
    private void addRegistrationComponents(GridBagConstraints gbc) {
        registrationPanel.add(regUsernameLabel, gbc);
        gbc.gridy++;
        gbc.insets = new Insets(20, 20, 0, 0);
        registrationPanel.add(emailLabel, gbc);
        gbc.gridy++;
        registrationPanel.add(phoneLabel, gbc);
        gbc.gridy++;
        registrationPanel.add(regPasswordLabel, gbc);
        gbc.gridy++;
        registrationPanel.add(confirmPasswordLabel, gbc);

        gbc.gridx++;
        gbc.gridy = 0;
        gbc.gridwidth = 2;
        gbc.fill = GridBagConstraints.HORIZONTAL;
        gbc.weightx = 1;
        gbc.insets = new Insets(10, 10, 0, 20);
        registrationPanel.add(regUsernameField, gbc);
        gbc.gridy++;
        gbc.insets = new Insets(20, 10, 0, 20);
        registrationPanel.add(emailField, gbc);
        gbc.gridy++;
        registrationPanel.add(phoneField, gbc);
        gbc.gridy++;
        registrationPanel.add(regPasswordField, gbc);
        gbc.gridy++;
        registrationPanel.add(confirmPasswordField, gbc);

        gbc.gridx = 1;
        gbc.gridy = 6;
        gbc.weightx = 0;
        gbc.gridwidth = 1;
        gbc.insets = new Insets(20, 35, 10, 20);
        gbc.fill = GridBagConstraints.NONE;
        registrationPanel.add(registerSubmitButton, gbc);
        gbc.gridx++;
        registrationPanel.add(regCancelButton, gbc);
    }

    /**
     * Handles button click events from the UI.
     * Routes each action to its appropriate handler method.
     *
     * @param e the action event containing the source button
     */
    @Override
    public void actionPerformed(ActionEvent e) {
        Object source = e.getSource();
        if (source == loginButton) {
            handleLogin();
        } else if (source == cancelButton) {
            System.exit(0);
        } else if (source == registerButton) {
            switchToRegistration();
        } else if (source == registerSubmitButton) {
            handleRegistration();
        } else if (source == regCancelButton) {
            switchToLogin();
        }
    }

    /**
     * Handles the login button click event.
     * Validates input and attempts to authenticate the user.
     */
    private void handleLogin() {
        if (!validateLoginInput()) {
            return;
        }

        String username = usernameField.getText();
        String password = new String(passwordField.getPassword());
        
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
     * Validates the login form input fields.
     * Checks that required fields are not empty.
     *
     * @return true if all required fields are filled, false otherwise
     */
    private boolean validateLoginInput() {
        if (usernameField.getText().isEmpty()) {
            JOptionPane.showMessageDialog(frame, "Please enter Username!");
            return false;
        }
        if (passwordField.getPassword().length == 0) {
            JOptionPane.showMessageDialog(frame, "Please enter Password!");
            return false;
        }
        return true;
    }

    /**
     * Attempts to authenticate the user against the database.
     *
     * @param username the username to authenticate
     * @param password the password to verify
     * @return true if authentication succeeds, false otherwise
     * @throws SQLException if there is an error accessing the database
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
     * Attempts to authenticate the user against the local file.
     *
     * @param username the username to authenticate
     * @param password the password to verify
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
     * Creates and displays the face recognition interface.
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
     * Switches the display from login to registration panel.
     * Removes the login panel and adds the registration panel.
     */
    private void switchToRegistration() {
        remove(loginPanel);
        add(registrationPanel);
        repaint();
        revalidate();
    }

    /**
     * Switches the display from registration to login panel.
     * Removes the registration panel and adds the login panel.
     */
    private void switchToLogin() {
        remove(registrationPanel);
        add(loginPanel);
        repaint();
        revalidate();
    }

    /**
     * Handles the registration button click event.
     * Validates input and attempts to register the new user.
     */
    private void handleRegistration() {
        if (!validateRegistrationInput()) {
            return;
        }

        String username = regUsernameField.getText();
        String email = emailField.getText();
        String phone = phoneField.getText();
        String password = new String(regPasswordField.getPassword());

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
     * Validates the registration form input fields.
     * Checks that all required fields are filled and properly formatted.
     *
     * @return true if all fields are valid, false otherwise
     */
    private boolean validateRegistrationInput() {
        if (regUsernameField.getText().isEmpty()) {
            JOptionPane.showMessageDialog(frame, "Please enter Username!");
            return false;
        }
        if (regUsernameField.getText().length() > 30) {
            JOptionPane.showMessageDialog(frame, "Maximum 30 characters allowed for Username!");
            return false;
        }
        if (emailField.getText().isEmpty()) {
            JOptionPane.showMessageDialog(frame, "Please enter Email!");
            return false;
        }
        if (!isValidEmail(emailField.getText())) {
            JOptionPane.showMessageDialog(frame, "Invalid Email!");
            return false;
        }
        if (phoneField.getText().isEmpty()) {
            JOptionPane.showMessageDialog(frame, "Please enter Phone!");
            return false;
        }
        if (!isValidPhone(phoneField.getText())) {
            JOptionPane.showMessageDialog(frame, "Invalid Phone!");
            return false;
        }
        if (regPasswordField.getPassword().length == 0) {
            JOptionPane.showMessageDialog(frame, "Please enter Password!");
            return false;
        }
        if (regPasswordField.getPassword().length < 3 || regPasswordField.getPassword().length > 15) {
            JOptionPane.showMessageDialog(frame, 
                "Password should be minimum 3 characters and maximum 15 characters!");
            return false;
        }
        if (confirmPasswordField.getPassword().length == 0) {
            JOptionPane.showMessageDialog(frame, "Please confirm password!");
            return false;
        }
        if (!new String(regPasswordField.getPassword()).equals(new String(confirmPasswordField.getPassword()))) {
            JOptionPane.showMessageDialog(frame, "Password Mismatch: Check again!");
            return false;
        }
        return true;
    }

    /**
     * Validates an email address format.
     *
     * @param email the email address to validate
     * @return true if the email format is valid, false otherwise
     */
    private boolean isValidEmail(String email) {
        String emailPattern = "^[_A-Za-z0-9-]+(\\.[_A-Za-z0-9-]+)*@[A-Za-z0-9-]+" +
            "(\\.[A-Za-z0-9-]+)*(\\.[A-Za-z]{2,})$";
        return email.matches(emailPattern);
    }

    /**
     * Validates a phone number format.
     *
     * @param phone the phone number to validate
     * @return true if the phone format is valid, false otherwise
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
     * Attempts to register a new user in the database.
     *
     * @param username the username for the new account
     * @param email the email address for the new account
     * @param phone the phone number for the new account
     * @param password the password for the new account
     * @return true if registration succeeds, false otherwise
     * @throws SQLException if there is an error accessing the database
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
     * Attempts to register a new user in the local file.
     *
     * @param username the username for the new account
     * @param password the password for the new account
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
     * Handles registration errors by displaying appropriate messages.
     *
     * @param ex the exception that occurred during registration
     */
    private void handleRegistrationError(Exception ex) {
        JOptionPane.showMessageDialog(frame, "Registration failed!");
    }

    /**
     * Hashes a password using SHA-256 algorithm.
     *
     * @param password the password to hash
     * @return the hashed password as a Base64 encoded string
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
     * Creates and displays the main application window.
     * Sets up the frame and initializes the UI components.
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
     * Main entry point for the application.
     * Launches the login interface.
     *
     * @param args command line arguments (not used)
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
}
