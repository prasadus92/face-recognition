//Author: Prasad U S
//Description: Front end for the project-3D Face Recognition
//Fun work: Usage of same frame for different inner layouts
//Dependencies: MySQL Driver
package src;

import java.awt.*;
import java.awt.event.*;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.PrintWriter;
import java.sql.*;
import java.lang.Long;
import javax.swing.*;
import javax.swing.border.TitledBorder;

public class FrontEnd extends JPanel implements ActionListener {

    public static JFrame frame = new JFrame("3D Face Recognition under Expressions, Occlusions and Pose Variations"); //This should be static
    JPanel firstPanel = new JPanel();
    JPanel secondPanel = new JPanel();
    //Needed for login page
    JLabel uname = new JLabel("Username*:");
    JLabel pwd = new JLabel("Password*:");
    JTextField un = new JTextField(30);
    JPasswordField pd = new JPasswordField(20);
    JButton login = new JButton("Login");
    JButton cancel = new JButton("Cancel");
    JButton register = new JButton("Click Here!");
    //Needed for registration page
    JLabel uname1 = new JLabel("Username*:");
    JLabel eml1 = new JLabel("Email*:");
    JLabel ph1 = new JLabel("Phone*:");
    JLabel pwd1 = new JLabel("Password*:");
    JLabel cpwd1 = new JLabel("Confirm Password*:");
    JTextField un1 = new JTextField(30);
    JTextField email1 = new JTextField(20);
    JTextField pn1 = new JTextField(20);
    JPasswordField pd1 = new JPasswordField(20);
    JPasswordField cpd1 = new JPasswordField(20);
    JButton reg1 = new JButton("Register");
    JButton cancel1 = new JButton("Cancel");

    public FrontEnd() {
        super(new GridBagLayout());		//Nurturing the flexibility and simplicity of GridBagLayout
        GridBagLayout gb1 = new GridBagLayout();
        GridBagLayout gb2 = new GridBagLayout();

        //Everything for login   	

        uname.setFont(new Font("Calibri", Font.BOLD, 16));  //Fun with the Calibri font, hope it looks nice
        pwd.setFont(new Font("Calibri", Font.BOLD, 16));

        firstPanel.setLayout(gb1);
        firstPanel.setBorder(new TitledBorder("Login"));

        GridBagConstraints gbc = new GridBagConstraints();
        gbc.anchor = GridBagConstraints.WEST;		//Align all components left side
        gbc.gridx = 0;
        gbc.gridy = 0;
        //(gridx,gridy)=(0,0) indicates top left corner
        gbc.insets = new Insets(10, 20, 0, 0);

        gb1.setConstraints(uname, gbc);
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
        login.addActionListener(this);
        gbc.gridx++;
        firstPanel.add(cancel, gbc);
        cancel.addActionListener(this);

        gbc.gridx = 1;
        gbc.gridy++;
        gbc.weighty = 0;
        gbc.gridwidth = 1;
        gbc.insets = new Insets(30, 0, 10, 20);
        gbc.fill = GridBagConstraints.HORIZONTAL;
        firstPanel.add(new JLabel("Dont have an account?"), gbc);
        gbc.gridx++;
        gbc.insets = new Insets(30, 0, 10, 20);
        firstPanel.add(register, gbc);
        register.addActionListener(this);
        //Login Ends



        //Register Page Layout begins

        uname1.setFont(new Font("Calibri", Font.BOLD, 16));
        eml1.setFont(new Font("Calibri", Font.BOLD, 16));
        ph1.setFont(new Font("Calibri", Font.BOLD, 16));
        pwd1.setFont(new Font("Calibri", Font.BOLD, 16));
        cpwd1.setFont(new Font("Calibri", Font.BOLD, 16));

        secondPanel.setLayout(gb2);
        secondPanel.setBorder(new TitledBorder("Register"));

        GridBagConstraints gbc1 = new GridBagConstraints();
        gbc1.gridx = 0;
        gbc1.gridy = 0;

        gbc1.insets = new Insets(10, 20, 0, 0);
        gbc1.anchor = GridBagConstraints.WEST;
        gb1.setConstraints(uname1, gbc1);
        secondPanel.add(uname1, gbc1);
        gbc1.gridy++;


        gbc1.insets = new Insets(20, 20, 0, 0);
        secondPanel.add(eml1, gbc1);
        gbc1.gridy++;


        gbc1.insets = new Insets(20, 20, 0, 0);

        secondPanel.add(ph1, gbc1);
        gbc1.gridy++;


        gbc1.insets = new Insets(20, 20, 0, 0);

        secondPanel.add(pwd1, gbc1);
        gbc1.gridy++;


        gbc1.insets = new Insets(20, 20, 0, 0);

        secondPanel.add(cpwd1, gbc1);

        gbc1.gridx++;
        gbc1.gridy = 0;
        gbc1.gridwidth = 2;
        gbc1.fill = GridBagConstraints.HORIZONTAL;
        gbc1.weightx = 1;

        gbc1.insets = new Insets(10, 10, 0, 20);

        secondPanel.add(un1, gbc1);
        gbc1.gridy++;

        gbc1.insets = new Insets(20, 10, 0, 20);

        secondPanel.add(email1, gbc1);
        gbc1.gridy++;

        gbc1.insets = new Insets(20, 10, 0, 20);

        secondPanel.add(pn1, gbc1);
        gbc1.gridy++;


        gbc1.insets = new Insets(20, 10, 0, 20);

        secondPanel.add(pd1, gbc1);
        gbc1.gridy++;


        gbc1.insets = new Insets(20, 10, 0, 20);

        secondPanel.add(cpd1, gbc1);

        gbc1.gridx = 1;
        gbc1.gridy = 6;
        gbc1.weightx = 0;
        gbc1.gridwidth = 1;
        gbc1.insets = new Insets(20, 35, 10, 20);
        gbc1.fill = GridBagConstraints.NONE;
        secondPanel.add(reg1, gbc1);
        reg1.addActionListener(this);
        gbc1.gridx++;
        secondPanel.add(cancel1, gbc1);
        cancel1.addActionListener(this);
        //Register page panel layout ends         


        add(firstPanel);   //Frame by default starts with login page
    }

    public void actionPerformed(ActionEvent e) {

        if (e.getSource().equals(login)) {
            if (un.getText().equals("")) {
                JOptionPane.showMessageDialog(frame, "Please enter Username!");
                return;
            }
            if (pd.getText().equals("")) {
                JOptionPane.showMessageDialog(frame, "Please enter Password!");
                return;
            }
            String username = un.getText();

            String password = pd.getText();
            int dbconn = 0;

            try {
                Class.forName("com.mysql.jdbc.Driver");
                Connection con = DriverManager.getConnection("jdbc:mysql://localhost/3dface", "root", "");
                dbconn = 1;
                String query = "select UserName,Password from users";
                Statement st = con.createStatement();
                ResultSet res = st.executeQuery(query);

                int success = 0;
                while (res.next() == true) {

                    if (username.equals(res.getString("UserName")) && password.equals(res.getString("Password"))) {

                        success = 1;
                        //Successful Login

                        JFrame j2 = new JFrame("3D Face Recognition under Expressions, Occlusions and Pose Variations");

                        JApplet applet = new Main();
                        j2.add(applet);
                        j2.setLocation(0, 0);
                        j2.setSize(1366, 730);
                        j2.setVisible(true);
                        applet.init();
                        //j2.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                        frame.dispose();

                        j2.addWindowListener(new java.awt.event.WindowAdapter() {
                            @Override
                            public void windowClosing(java.awt.event.WindowEvent windowEvent) {
                                int reply;
                                if ((reply = JOptionPane.showConfirmDialog(frame,
                                        "Are you sure you want to exit?", "Confirmation",
                                        JOptionPane.YES_NO_OPTION)) == JOptionPane.YES_OPTION) {
                                    System.exit(0);
                                }
                            }
                        });
                        j2.setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
                        j2.setVisible(true);
                        break;
                    }
                }
                if (success == 0) {
                    JOptionPane.showMessageDialog(frame, "Login failed: Check Details!");
                }
                con.close();   //Database connection closed for data privacy
                //return;
            } catch (Exception ex) {
                if (dbconn == 0) {
                    try {
                        BufferedReader reader = new BufferedReader(new FileReader("users.txt"));
                        String line = null;
                        line = reader.readLine();

                        String[] parts = line.split("\\s");
                        if (username.equals(parts[0]) && password.equals(parts[1])) {


                            JFrame j2 = new JFrame("3D Face Recognition under Expressions, Occlusions and Pose Variations");

                            JApplet applet = new Main();
                            j2.add(applet);
                            j2.setLocation(0, 0);
                            j2.setSize(1366, 730);
                            j2.setVisible(true);
                            applet.init();
                            //j2.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
                            frame.dispose();

                            j2.addWindowListener(new java.awt.event.WindowAdapter() {
                                @Override
                                public void windowClosing(java.awt.event.WindowEvent windowEvent) {
                                    int reply;
                                    if ((reply = JOptionPane.showConfirmDialog(frame,
                                            "Are you sure you want to exit?", "Confirmation",
                                            JOptionPane.YES_NO_OPTION)) == JOptionPane.YES_OPTION) {
                                        System.exit(0);
                                    }
                                }
                            });
                            j2.setDefaultCloseOperation(WindowConstants.DO_NOTHING_ON_CLOSE);
                            j2.setVisible(true);


                        }

                    } catch (Exception eeeeee) {
                    }
                } else {
                    JOptionPane.showMessageDialog(frame, "Login failed!");
                }
                //return;
            }
        } else if (e.getSource().equals(cancel)) {
            System.exit(0);
        } else if (e.getSource() == register) {
            remove(firstPanel);
            add(secondPanel);
            repaint();
            revalidate();
        } else if (e.getSource() == reg1) {	//Code for registration(Database insertion)
            if (un1.getText().equals("")) {
                JOptionPane.showMessageDialog(frame, "Please enter Username!");
                return;
            }
            if (!(un1.getText().equals("")) && un1.getText().length() > 30) {
                JOptionPane.showMessageDialog(frame, "Maximum 30 characters allowed for Username!");
                return;
            }
            if (email1.getText().equals("")) {
                JOptionPane.showMessageDialog(frame, "Please enter Email!");
                return;
            }
            if (!(email1.getText().equals(""))) {
                String emailPattern = "^[_A-Za-z0-9-]+(\\.[_A-Za-z0-9-]+)*@[A-Za-z0-9-]+(\\.[A-Za-z0-9-]+)*(\\.[A-Za-z]{2,})$";
                if (!(email1.getText().matches(emailPattern))) {
                    JOptionPane.showMessageDialog(frame, "Invalid Email!");
                    return;
                }
            }
            if (pn1.getText().equals("")) {
                JOptionPane.showMessageDialog(frame, "Please enter Phone!");
                return;
            }
            if (!(pn1.getText().equals(""))) {
                try {
                    long p = 9999999999L;				//Long values should be appended with L or l at the end
                    long r = 1000000000L;
                    long q = Long.parseLong(pn1.getText(), 10);		//parseLong(string,radix), for decimal radix=10
                    if (q > p || q < r) {			//Checking for 10 digits
                        JOptionPane.showMessageDialog(frame, "Invalid Phone!");
                        return;
                    }
                } catch (Exception ep) {
                    JOptionPane.showMessageDialog(frame, "Invalid Phone!");
                    return;
                }
            }
            if (pd1.getText().equals("")) {
                JOptionPane.showMessageDialog(frame, "Please enter Password!");
                return;
            }
            if (!(pd1.getText().equals("")) && ((pd1.getText().length() < 3) || (pd1.getText().length() > 15))) {
                JOptionPane.showMessageDialog(frame, "Password should be minimum 3 characters and maximum 15 characters!");
                return;
            }
            if (cpd1.getText().equals("")) {
                JOptionPane.showMessageDialog(frame, "Please confirm password!");
                return;
            }
            if (!(pd1.getText().equals("")) && !(cpd1.getText().equals(""))) {
                if (!(pd1.getText().equals(cpd1.getText()))) {
                    JOptionPane.showMessageDialog(frame, "Password Mismatch:Check again!");
                    return;
                }
            }

            //All input data valid, we can play with database now
            int exists = 0;
            int dcon = 0;
            int update = 0;
            String userName;
            String password;
            userName = un1.getText();
            String email = email1.getText();
            String phone = pn1.getText();
            password = pd1.getText();
            try {

                Class.forName("com.mysql.jdbc.Driver");
                Connection con1 = DriverManager.getConnection("jdbc:mysql://localhost/3dface", "root", "");
                dcon = 1;
                String query2 = "select Email from users";
                Statement st2 = con1.createStatement();
                ResultSet res2 = st2.executeQuery(query2);
                while (res2.next() == true) {
                    if (email.equals(res2.getString("Email"))) {
                        exists = 1; //Email already registered
                        Exception ec = new Exception();	//Raise an exception here
                        throw ec;
                    }
                }

                String query1 = "insert into users values('" + userName + "','" + email + "','" + phone + "','" + password + "')";
                Statement st1 = con1.createStatement();
                st1.executeUpdate(query1);
                update = 1;
                JOptionPane.showMessageDialog(frame, "Registration successful!");
                con1.close();  //Database connection closed
                remove(secondPanel);
                add(firstPanel);
                repaint();
                revalidate();
            } catch (Exception expo) {
                if (dcon == 0) {
                    try {
                        PrintWriter writer = new PrintWriter("users.txt", "UTF-8");
                        writer.println(userName + " " + password);
                        writer.close();
                        //System.out.println(userName);
                        JOptionPane.showMessageDialog(frame, "Registration successful!");
                    } catch (Exception eee) {
                        System.out.println(eee);
                    }
                    return;
                } else if (update == 0 && exists == 0) {
                    JOptionPane.showMessageDialog(frame, "Registration failed!");
                    return;
                } else if (exists == 1) {
                    JOptionPane.showMessageDialog(frame, "Email already registered!");
                    return;
                }
            }

        } else if (e.getSource().equals(cancel1)) {
            remove(secondPanel);
            add(firstPanel);
            repaint();
            revalidate();
        }
    }

    /**
     * Create the GUI and show it. For thread safety, this method should be
     * invoked from the event-dispatching thread.
     */
    private static void createAndShowGUI() {
        // Set up the window.
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        // Create and set up the content pane.
        JComponent newContentPane = new FrontEnd();
        newContentPane.setOpaque(true); // Content panes must be opaque

        frame.setContentPane(newContentPane);
        frame.setPreferredSize(new Dimension(700, 450));

        frame.setLocation(350, 150);
        frame.setResizable(false);
        frame.pack();				//Redundant usage here, because I already used setPreferredSize(), can be commented
        frame.setVisible(true);
    }

    public static void main(String[] args) {
        // Schedule a job for the event-dispatching thread:
        // Creating and showing this application's GUI.
        javax.swing.SwingUtilities.invokeLater(new Runnable() {
            public void run() {
                try {
                    UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
                } catch (Exception ex) {
                    ex.printStackTrace();
                }
                createAndShowGUI();
            }
        });
    }
}
