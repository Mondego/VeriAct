package escjava.gui;

import javax.swing.*;
import javax.swing.tree.*;
import javax.swing.event.*;
import java.awt.*;
import java.awt.event.*;
import java.io.File;
import javax.swing.filechooser.FileFilter;
import java.io.StringReader;
import java.io.BufferedReader;
import java.util.Iterator;
import java.util.ArrayList;
import java.util.Timer;
import java.util.TimerTask;

import escjava.Status;
import javafe.InputEntry;
import javafe.ast.CompilationUnit;

//import com.apple.eawt.*;

public class EscFrame extends JFrame {

  static public final boolean runningOnMac = (System.getProperty("mrj.version") != null);

  static public /*@non_null*/ String addDots(/*@non_null*/ String s) {
    if (!runningOnMac) return s + "...";
    return s;
  }

  static {
    // Checks to see if we are running on a Mac
    if (runningOnMac) {
      // If so, then puts the menubar at the top of the screen
      // instead of the top of the window
      System.setProperty("apple.laf.useScreenMenuBar","true");
    }
  }

  static public final /*@non_null*/ JMenuBar menubar = new JMenuBar();
  public /*@ non_null */ EscOptions escoptionPanel;
  public /*@ non_null */ GuiOptionsPanel guioptionPanel;
  private /*@ non_null */ JTextArea listArea;
  static /*@ non_null */ JLabel label;
  /*@ non_null */ JScrollPane treeView;
  /*@ non_null */ JLabel sizeinfo;
  /*@ non_null */ JComponent guilight;
  /*@ non_null */ JComponent proverlight;
  static /*@ non_null */ JTree tree;

  /*@ non_null */ JTextField currentdirText;
  /*@ non_null */ JTextField classpathText, specspathText;
  /*@ non_null */ JFileChooser fc = new JFileChooser(GUI.gui.options.currentdir);

  public EscFrame() {
    setTitle("ESC/Java2 (Version: " + escjava.Version.VERSION+ ")");
    setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    buildGUI(this);

    pack();

    EscHtml jf = EscHtml.make("Welcome","escjava/gui/welcome.html",
        this,100,100,500,500);

    FrameShower.show(this);

    jf.showit();

  }

  private /*@ helper */ void buildGUI(final /*@non_null*/ JFrame frame) {
    JTabbedPane tabbedPane = new JTabbedPane();
    frame.getContentPane().add(tabbedPane, BorderLayout.CENTER);
    guioptionPanel = new GuiOptionsPanel();
    tabbedPane.addTab("GUI Options",null,guioptionPanel,
    "Options for controlling the User Interface");

    escoptionPanel = new EscOptions(GUI.options());
    tabbedPane.addTab("ESC Options", null, escoptionPanel,
    "Options for controlling the static checking tool");

    JPanel filesPanel = new JPanel(new BorderLayout());
    tabbedPane.addTab("Project files", null, filesPanel,
    "Tools to control the files used in this project");
    JPanel topFilesPanel = new JPanel(new GridLayout(0,1));
    filesPanel.add(topFilesPanel,BorderLayout.NORTH);
    /* Changing the user.dir during a program appears to be problematic - at
   least on a Mac, new File instances do not appear to use the new value.

	topFilesPanel.add(new JLabel("Current directory:"));
	topFilesPanel.add(currentdirText = new JTextField(escjava.Main.options.currentdir));
	currentdirText.addActionListener( new ActionListener() {
	    public void actionPerformed(ActionEvent e) {
		String s = currentdirText.getText();
		if (s.equals(escjava.Main.options.currentdir)) return;
		escjava.Main.options.currentdir = s;
		System.setProperty("user.dir",s);
		GUI.gui.restart(null);
		label.setText("Clearing all results because the Working Directory changed");
	    }
	});
	currentdirText.addFocusListener( new FocusAdapter() {
	    public void focusLost(FocusEvent e) {
		String s = currentdirText.getText();
		if (s.equals(escjava.Main.options.currentdir)) return;
		escjava.Main.options.currentdir = s;
		System.setProperty("user.dir",s);
		GUI.gui.restart(null);
		label.setText("Clearing all results because the Working Directory changed");
	    }
	});
     */
    topFilesPanel.add(new JLabel("CLASSPATH:"));
    topFilesPanel.add(classpathText = new JTextField(escjava.Main.options.userPath,50));
    classpathText.addActionListener( new ActionListener() {
      public void actionPerformed(/*@non_null*/ActionEvent e) {
        String s = classpathText.getText();
        if (s.equals(escjava.Main.options.userPath)) return;
        escjava.Main.options.userPath = s;
        GUI.gui.restart(null);
        label.setText("Clearing all results because the CLASSPATH changed");
      }
    });
    classpathText.addFocusListener( new FocusAdapter() {
      public void focusLost(/*@ non_null */ FocusEvent e) {
        String s = classpathText.getText();
        if (s.equals(escjava.Main.options.userPath)) return;
        escjava.Main.options.userPath = s;
        GUI.gui.restart(null);
        label.setText("Clearing all results because the CLASSPATH changed");
      }
    });

    topFilesPanel.add(new JLabel("Specs path:"));
    topFilesPanel.add(specspathText = new JTextField(escjava.Main.options().specspath,50));
    specspathText.addActionListener( new ActionListener() {
      public void actionPerformed(/*@non_null*/ActionEvent e) {
        String s = specspathText.getText();
        if (s.equals(escjava.Main.options().specspath)) return;
        escjava.Main.options().specspath = s;
        GUI.gui.restart(null);
        label.setText("Clearing all results because the Specs path changed");
      }
    });
    specspathText.addFocusListener( new FocusAdapter() {
      public void focusLost(/*@ non_null */ FocusEvent e) {
        String s = specspathText.getText();
        if (s.equals(GUI.gui.options().specspath)) return;
        GUI.gui.options().specspath = s;
        GUI.gui.restart(null);
        label.setText("Clearing all results because the Specs path changed");
      }
    });



    topFilesPanel.add(new JLabel("Files, directories, classes, packages and lists to process:"));
    listArea = new JTextArea();
    filesPanel.add(new JScrollPane(listArea),BorderLayout.CENTER);
    //listArea.getParent().setPreferredSize(new Dimension(300,200));
    loadTextArea();
    listArea.addFocusListener(
        new FocusAdapter() {
          public void focusLost(/*@ non_null */ FocusEvent e) {
            if (saveTextArea()) {
              GUI.gui.restart(null);
              label.setText("Clearing all results because the input changed");
            }
          }
        });


    JPanel panel = new JPanel(new GridLayout(2,1));

    JPanel buttonPanel = new JPanel(new FlowLayout());
    panel.add(buttonPanel);

    JButton button = new JButton("Reload");
    buttonPanel.add(button);
    button.addActionListener(new ActionListener() {
      public void actionPerformed(/*@non_null*/ActionEvent e) { 
        GUI.processTasks.addTask(new Integer(GUI.RELOAD));
      } });

    button = new JButton("Clear");
    buttonPanel.add(button);
    button.addActionListener(new ActionListener() {
      public void actionPerformed(/*@non_null*/ActionEvent e) { escAction(GUI.CLEAR); } });

    button = new JButton("Check");
    button.addActionListener(new ActionListener() {
      public void actionPerformed(/*@non_null*/ActionEvent e) { escAction(GUI.CHECK); } });
    buttonPanel.add(button);

    button = new JButton("Stop");
    buttonPanel.add(button);
    button.addActionListener(new ActionListener() {
      public void actionPerformed(/*@non_null*/ActionEvent e) { 
        synchronized (GUI.processTasks) {
          GUI.processTasks.clear();
          GUI.gui.stop = true; 
          escjava.ProverManager.kill();
          GUI.processTasks.addTask(GUI.STOP);
          //System.out.println("REQUESTED STOP");
        }
      } });

    JPanel infopanel = new JPanel();
    infopanel.setLayout(new BoxLayout(infopanel,BoxLayout.LINE_AXIS));
    panel.add(infopanel);

    label = new JLabel(" ");
    infopanel.add(label);
    frame.getContentPane().add(panel, BorderLayout.NORTH);

    infopanel.add(Box.createHorizontalGlue());
    sizeinfo = new JLabel(" ");
    infopanel.add(sizeinfo);
    updateSizeInfo();
    TimerTask tt = new TimerTask() {
      public void run() { updateSizeInfo(); }
    };
    Timer ttt = new Timer(true);
    ttt.schedule(tt,0,1000);

    guilight = new JPanel();
    Dimension d = new Dimension(10,10);
    guilight.setMaximumSize(d);
    guilight.setMinimumSize(d);
    guilight.setToolTipText("Shows the status of the GUI:" +
        "Blue - waiting for processing commands; " +
        "Green - parsing, typechecking, VC generation; " +
    "Yellow - waiting for the prover");
    infopanel.add(guilight);

    proverlight = new JPanel();
    proverlight.setMaximumSize(d);
    proverlight.setMinimumSize(d);
    proverlight.setToolTipText("Shows the status of the Prover:" +
        "Blue - waiting for VCs to check; " +
        "Green - proving; " +
    "Black - prover is not executing");
    infopanel.add(proverlight);

    showGuiLight(0);
    showProverLight(0);
    escjava.ProverManager.listener = new escjava.ProverManager.Listener() {
      public void stateChanged(int i) {
        showProverLight(i);
        if (i != 0) showGuiLight(3-i);
      }};

      tree = new JTree(GUI.gui.treeModel);
      tree.getSelectionModel().setSelectionMode(TreeSelectionModel.DISCONTIGUOUS_TREE_SELECTION);
      if (guioptionPanel.settings.autoExpand) {
        int k=GUI.gui.treeModel.getChildCount(GUI.gui.treeModel.getRoot());
        while (k > 0) tree.expandRow(--k);
      }
      DefaultTreeCellRenderer r = new EscRenderer();
      r.setLeafIcon(null);
      r.setOpenIcon(null);
      r.setClosedIcon(null);
      /*
	TreeCellRenderer r = new EscTreeCellRenderer();
       */
      tree.addMouseListener(
          new MouseInputAdapter() {
            //@ also
            //@    requires e != null;
            public void mouseClicked(/*@ non_null */ MouseEvent e) {
              if (e.getClickCount() != 1) return;
              Object o = ((DefaultMutableTreeNode)
                  tree.getClosestPathForLocation(e.getX(),e.getY()).
                  getLastPathComponent()).getUserObject();
              if ((e.getModifiers() & MouseEvent.ALT_MASK) != 0)
                // Alt + click --> error window
                WindowThread.windowTasks.addTask(o);
              else if ((e.getModifiers() & MouseEvent.CTRL_MASK) != 0) {
                // Ctrl + click --> editor window
                if (o instanceof GUI.EscTreeValue) {
                  showEditor((GUI.EscTreeValue)o);
                }
              }
            }}); 
      tree.setCellRenderer(r);

      tree.putClientProperty("JTree.lineStyle", "Angled");
      tree.setShowsRootHandles(true);
      ToolTipManager.sharedInstance().registerComponent(tree);
      tree.setRootVisible(false);
      //tree.setEditable(true);
      //tree.setDragEnabled(true);
      treeView = new JScrollPane(tree);
      //treeView.setPreferredSize(new Dimension(400,300));
      treeView.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);
      tabbedPane.addTab("Results", null, treeView,
      "Results of checking the project files");
      // Default tab is Project tab if there are no files set,
      // otherwise it is the Results tab
      int def = 3;
      if (listArea.getText().length() == 0) def = 2;
      tabbedPane.setSelectedIndex(def);

      // Menus

      frame.setJMenuBar(menubar);
      JMenu menu = new JMenu("File");
      menubar.add(menu);

      JMenuItem mi;
      menu.add(mi = new JMenuItem("New Project"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              Project.init();
            }
          });

      menu.add(mi = new JMenuItem(addDots("Open Project")));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              fc.addChoosableFileFilter(
                  new FileFilter() {
                    public boolean accept(/*@ non_null */ File f) {
                      return !f.isFile() || f.getName().endsWith(".esc");
                    }
                    public /*@ non_null */ String getDescription() {
                      return "*.esc";
                    }
                  });
              int returnVal = fc.showOpenDialog(frame);
              if (returnVal == JFileChooser.APPROVE_OPTION) {
                File file = fc.getSelectedFile();
                Project.read(file);
                escoptionPanel.init(escjava.Main.options());
              }
            }
          });

      menu.add(mi = new JMenuItem("Save Project"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              if (Project.currentProject != null)
                Project.write();
              else {
                int returnVal = fc.showSaveDialog(frame);
                if (returnVal == JFileChooser.APPROVE_OPTION) {
                  File file = fc.getSelectedFile();
                  if (file.exists()) {
                    boolean b = JOptionPane.showConfirmDialog(
                        EscFrame.this,
                        "The file " + file.getAbsolutePath() +
                        " already exists - do you want to overwrite it?",
                        "Confirm Save",JOptionPane.OK_CANCEL_OPTION)
                        == JOptionPane.OK_OPTION;
                    if (!b) return;
                  }
                  Project.write(file);
                }
              }
            }
          });

      menu.add(mi = new JMenuItem(addDots("SaveAs Project")));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              int returnVal = fc.showSaveDialog(frame);
              if (returnVal == JFileChooser.APPROVE_OPTION) {
                File file = fc.getSelectedFile();
                if (file.exists()) {
                  boolean b = JOptionPane.showConfirmDialog(
                      EscFrame.this,
                      "The file " + file.getAbsolutePath() +
                      " already exists - do you want to overwrite it?",
                      "Confirm Save",JOptionPane.OK_CANCEL_OPTION)
                      == JOptionPane.OK_OPTION;
                  if (!b) return;
                }
                Project.write(file);
              }
            }
          });

      if (!runningOnMac) {
        menu.add(mi = new JMenuItem("Exit"));
        mi.addActionListener( new ActionListener() {
          public void actionPerformed(/*@non_null*/ActionEvent e) { System.exit(0); }
        } );
      }

      menubar.add(menu = new JMenu("View"));
      menu.add(mi = new JMenuItem("Error window"));
      mi.addActionListener( new ActionListener() {
        public void actionPerformed(/*@non_null*/ActionEvent e) {
          TreePath[] paths = tree.getSelectionPaths();
          for (int i=0; i<paths.length; ++i) {
            Object o = ((DefaultMutableTreeNode)paths[i].
                getLastPathComponent()).
                getUserObject();
            WindowThread.windowTasks.addTask(o);
          }
        }
      });
      menu.add(mi = new JMenuItem("Editor window"));
      mi.addActionListener( new ActionListener() {
        public void actionPerformed(/*@non_null*/ActionEvent e) {
          TreePath[] paths = tree.getSelectionPaths();
          for (int i=0; i<paths.length; ++i) {
            Object o = ((DefaultMutableTreeNode)paths[i].
                getLastPathComponent()).
                getUserObject();
            GUI.EscTreeValue v = (GUI.EscTreeValue)o;
            showEditor(v);
          }
        }
      } );



      menubar.add(menu = new JMenu("Check"));

      menu.add(mi = new JMenuItem("Check All"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              GUI.processTasks.addTask(new Integer(GUI.CHECK));
            }});

      menu.add(mi = new JMenuItem("Check Selected"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              if (!tree.isSelectionEmpty()) escAction(GUI.CHECK);
            }});

      menu.add(mi = new JMenuItem("TypeCheck All"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              GUI.processTasks.addTask(new Integer(GUI.TYPECHECK));
            }});

      menu.add(mi = new JMenuItem("TypeCheck Selected"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              if (!tree.isSelectionEmpty()) escAction(GUI.TYPECHECK);
            }});

      menu.add(mi = new JMenuItem("Parse All"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              GUI.processTasks.addTask(new Integer(GUI.PARSE));
            }});

      menu.add(mi = new JMenuItem("Parse Selected"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              if (!tree.isSelectionEmpty()) escAction(GUI.PARSE);
            }});

      menu.add(mi = new JMenuItem("Resolve All"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              GUI.processTasks.addTask(new Integer(GUI.RESOLVE));
            }});

      menu.add(mi = new JMenuItem("Resolve Selected"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              if (!tree.isSelectionEmpty()) escAction(GUI.RESOLVE);
            }});

      menu.add(mi = new JMenuItem("Clear All"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              GUI.processTasks.addTask(new Integer(GUI.CLEAR));
            }});

      menu.add(mi = new JMenuItem("Clear Selected"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              if (!tree.isSelectionEmpty()) escAction(GUI.CLEAR);
            }});

      menubar.add(menu = new JMenu("Tools"));
      menu.add(mi = new JMenuItem("Gen. Skeleton"));
      mi.addActionListener( notimp(frame) );

      menu.add(mi = new JMenuItem("Garbage Collect"));
      mi.addActionListener( new ActionListener() {
        public void actionPerformed(/*@non_null*/ActionEvent e) {
          System.gc();
        }
      });

      menubar.add(menu = new JMenu("L&F"));
      UIManager.LookAndFeelInfo[] looks = UIManager.getInstalledLookAndFeels();
      try {
        UIManager.setLookAndFeel(
            UIManager.getSystemLookAndFeelClassName());
      } catch (Exception eee) {}  // FIXME
      LookAndFeel current = UIManager.getLookAndFeel();
      String name = current.getClass().toString().substring(6); // 6 is the length of "class "
      ButtonGroup bg = new ButtonGroup();
      LAF milaf;
      for (int i=0; i<looks.length; ++i) {
        menu.add(milaf = new LAF(looks[i]));
        milaf.addActionListener(milaf);
        if (name.equals(looks[i].getClassName())) {
          milaf.setSelected(true);
        }
        bg.add(milaf);
      }

      menubar.add(menu = new JMenu("Help"));
      /*
	menu.add(mi = new JMenuItem("Usage"));
	mi.addActionListener(
	    new ActionListener() {
		public void actionPerformed(ActionEvent e) {
		    new EscOutputFrame("Escjava2 usage","USAGE");
		}
	    });
       */

      menu.add(mi = new JMenuItem("Documentation"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              WindowThread.windowTasks.addTask(new WindowThread.HtmlTask("Documentation",
              "escjava/gui/escjava2gui.html"));
            }
          });

      menu.add(mi = new JMenuItem("Issues/Bugs"));
      mi.addActionListener(
          new ActionListener() {
            public void actionPerformed(/*@non_null*/ActionEvent e) {
              WindowThread.windowTasks.addTask(new WindowThread.HtmlTask(
                  "Issues & Notes","escjava/gui/issues.html"));
            }
          });

  }

  static private class LAF extends JRadioButtonMenuItem implements ActionListener {
    public /*@ non_null */ UIManager.LookAndFeelInfo laf;
    public LAF(/*@non_null*/ UIManager.LookAndFeelInfo laf) {
      super(laf.getName());
      this.laf = laf;
    }
    public void actionPerformed(/*@non_null*/ActionEvent e) {
      try {
        UIManager.setLookAndFeel(laf.getClassName());
        SwingUtilities.updateComponentTreeUI(GUI.gui.escframe);
      } catch (Exception ee) {
        JOptionPane.showMessageDialog(GUI.gui.escframe,"Could not find Look & Feel named " + laf.getName());
      }
    }
  }

  public void showEditor(/*@non_null*/ GUI.EscTreeValue v) {
    String vname = v.getFilename();
    if (vname == null) return;
    if (v instanceof GUI.GFCUTreeValue) {
      GUI.GFCUTreeValue vv = (GUI.GFCUTreeValue)v;
      CompilationUnit cu = vv.cu;
      if (cu != null && cu instanceof escjava.RefinementSequence) {
        ArrayList a = ((escjava.RefinementSequence)cu).refinements();
        Iterator i = a.iterator();
        while (i.hasNext()) {
          CompilationUnit rcu = (CompilationUnit)i.next();
          String name = rcu.sourceFile().getHumanName();
          if (!name.equals(vname))
            WindowThread.windowTasks.addTask(name + ":1:");
          // FIXME - would like other elements of the compilation
          // unit to be scrolled to the same method
        }
      }
    }

    WindowThread.windowTasks.addTask( vname + ":" + v.getLine() + ":");
  }

  public void init() {
    classpathText.setText(GUI.gui.options.userPath);
    loadTextArea();
    tree.setModel(GUI.gui.treeModel);
    GUI.gui.treeModel.reload();
  }

  static /*@non_null*/ ActionListener notimp(final /*@non_null*/ JFrame parent) {
    return new ActionListener() {
      public void actionPerformed(/*@non_null*/ActionEvent e) {
        //System.out.println("NOT IMPLEMENTED");
        JOptionPane.showMessageDialog(parent,"Menu item not implemented");
      }
    };
  }

  static public void checkAction() {
    if (tree.isSelectionEmpty()) {
      System.out.println("CHECKING EVERYTHING");
      GUI.processTasks.addTask(PROCESS_ALL);
    } else {
      TreePath[] paths = tree.getSelectionPaths();
      for (int i=0; i<paths.length; ++i) {
        GUI.processTasks.addTask(((DefaultMutableTreeNode)paths[i].getLastPathComponent()).
            getUserObject());
      }
    }
  }

  static public void escAction(int action) {
    if (tree.isSelectionEmpty()) {
      GUI.processTasks.addTask(new Integer(action));
    } else {
      TreePath[] paths = tree.getSelectionPaths();
      for (int i=0; i<paths.length; ++i) {
        Object o = ((DefaultMutableTreeNode)paths[i].
            getLastPathComponent()).
            getUserObject();
        GUI.EscTreeValue v = (GUI.EscTreeValue)o;
        v.action = action;
        GUI.processTasks.addTask(v);
      }
    }
  }



  static private class EscRenderer extends DefaultTreeCellRenderer {
    //@ also
    //@   requires value != null;
    public /*@ non_null */ Component getTreeCellRendererComponent(
        /*@ non_null */ JTree tree,
        /*@ non_null */ Object value,
        boolean sel,
        boolean expanded,
        boolean leaf,
        int row,
        boolean hasFocus) {

      super.getTreeCellRendererComponent(
          tree, value, sel,
          expanded, leaf, row,
          hasFocus);
      Object userValue = ((DefaultMutableTreeNode)value).getUserObject();
      if (userValue instanceof GUI.EscTreeValue) {
        GUI.EscTreeValue u = (GUI.EscTreeValue)userValue;
        Color c = Status.toColor(u.status);
        Color cc = c == null ? new Color(200,200,255) :
          new Color((c.getRed()+510)/3,(c.getGreen()+510)/3,
              (c.getBlue()+510)/3);
        if (c == null) c = Color.WHITE;

        setBackgroundNonSelectionColor(c);
        setBackgroundSelectionColor(cc);
        setToolTipText(u.getStatusText());
      }

      return this;
    }
  }

  private /*@ nullable */ String cachedText = null;
  public boolean saveTextArea() {
    String s = listArea.getText();
    if (s.equals(cachedText)) return false;
    cachedText = s;
    ArrayList opts = GUI.options().inputEntries;
    opts.clear();
    BufferedReader r = new BufferedReader(new StringReader(s));
    try {
      while ((s = r.readLine()) != null) {
        String type = null;
        if (s.startsWith("-")) {
          int p = s.indexOf(' ');
          if (p != -1) {
            type = s.substring(1,p);
            s = s.substring(p+1);
          }
        }
        s = s.trim();
        if (s.length() == 0) continue;
        if (s.charAt(0) == '"' && s.charAt(s.length()-1) == '"')
          s = s.substring(1,s.length()-1);
        InputEntry ie = (InputEntry.make(type,s));
        if (ie != null) opts.add(ie);
        else if (type != null) {
          JOptionPane.showMessageDialog(this,
              "An invalid type in the list of inputs: "
              + type);
        }
      }
    } catch (java.io.IOException e) {
      // Not quite sure why this could ever happen, since we are
      // simply reading from a StringReader, but we'll pop a dialog
      // just in case.
      JOptionPane.showMessageDialog(this,
          "An error happened interpreting the list of input sources: "
          + e);
    }
    return true;
  }

  public void loadTextArea() {
    listArea.setText("");
    Iterator ii = GUI.options().inputEntries.iterator();
    while (ii.hasNext()) {
      InputEntry s = (InputEntry)ii.next();
      if (!s.auto) {
        listArea.append("-" + s.typeOption() + " ");
      }
      boolean q = s.name.indexOf(' ') != -1;
      if (q) listArea.append("\"");
      listArea.append(s.name);
      if (q) listArea.append("\"");
      listArea.append(Project.eol);
    }
  }


  static final /*@non_null*/ Object PROCESS_ALL = new Object();

  public void updateSizeInfo() {
    Runtime rt = Runtime.getRuntime();
    long memUsedBytes = rt.totalMemory() - rt.freeMemory();
    sizeinfo.setText(memUsedBytes + " bytes");
  }

  public void showGuiLight(int i) {
    guilight.setBackground( i==0 ? Color.BLUE : i == 1 ? Color.YELLOW : Color.GREEN);
  }
  public void showProverLight(int i) {
    proverlight.setBackground(i == 0? Color.BLACK : i == 2? Color.GREEN : Color.BLUE);
  }

  public class EscTreeCellRenderer extends JPanel implements TreeCellRenderer {
    /*@non_null*/ JLabel label;
    /*@non_null*/ JCheckBox cb;

    public EscTreeCellRenderer() {
      setLayout(new BoxLayout(this,BoxLayout.LINE_AXIS));
      add(label = new JLabel());
      add(Box.createHorizontalGlue());
      add(cb = new JCheckBox("",true));
    }

    public void setText(/*@ non_null */ String s) {
      label.setText(s);
    }

    public boolean isShowing() { return true; }

    public /*@non_null*/ Component getTreeCellRendererComponent(/*@ non_null */ JTree tree,
        /*@ non_null */ Object value,
        boolean selected,
        boolean expanded,
        boolean leaf,
        int row,
        boolean hasFocus) {

      if (value instanceof DefaultMutableTreeNode) {
        Object o = ((DefaultMutableTreeNode)value).getUserObject();
        if (o instanceof GUI.EscTreeValue) {
          GUI.EscTreeValue u = (GUI.EscTreeValue)o;
          Color c = Status.toColor(u.status);
          Color cc = c == null ? new Color(200,200,255) :
            new Color((c.getRed()+510)/3,(c.getGreen()+510)/3,
                (c.getBlue()+510)/3);
          if (c == null) c = Color.WHITE;

          setBackground(selected ? cc : c);
          setToolTipText(u.getStatusText());
          setText(u.toString());
          //Dimension d1 = getMinimumSize();
          Dimension d2 = getPreferredSize();
          //Dimension d3 = getMaximumSize();
//        System.out.println("DIM " + d1.width + " " + d1.height + " " + d2.width + " " + d2.height + " " + d3.width + " " + d3.height);
          int w = treeView.getViewport().getWidth();
//        System.out.println("SZ " + w);
          if (w == 0) w = 500;
//        System.out.println("LOC " + getX() + " " + getY());
          setPreferredSize(new Dimension(w - 100,d2.height));
//        CellRendererPane cp = (CellRendererPane)getParent();
//        if (cp != null) cp.setSize(600,d2.height);
//        if (cp != null) System.out.println("PAR " + cp.getPreferredSize().width + " " + cp.getPreferredSize().height + " " + cp.getMaximumSize().width + " " + cp.getMaximumSize().height + " " + cp.getX() + " " + cp.getY());
        }
      }
      return this;

    }


  }

  /*
Still puts up the Java about box.
Need to make a nicer about box - and not a modal one.
Also make this cross platform.

    static About about = new About();

    static public class About extends Application {

        public About() {
	    addApplicationListener(new AboutH());
	}

    }
    static public class AboutH extends ApplicationAdapter {

	public void handleAbout(ApplicationEvent e) {
	    JOptionPane.showMessageDialog(GUI.gui.escframe,"ABOUT");
	}
	public void handleQuit(ApplicationEvent e) {
	    System.exit(0);
        }
    }
   */
}
