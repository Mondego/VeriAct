// COpyright 2004, David Cok

package escjava.gui;

import java.lang.reflect.Field;
import javax.swing.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class GuiOptionsPanel extends JPanel implements ActionListener {

    static public class Settings {
	public boolean autoExpand = true;
	public boolean autoScroll = true;
	public boolean breadthFirst = false;
	public boolean autoShowErrors = true;
    }

    public Settings settings = new Settings();

	/**  The array contains the text string, field name and the tooltip. */
    static final public String[][] info = {
	{ "Auto Expand the nodes", "autoExpand", "Automatically expands the nodes as processing progresses"},
	{ "Auto scroll", "autoScroll", "Automatically scroll the window to keep processing point in view (will also expand nodes)"},
	{ "Breadth first checking", "breadthFirst", "Check all nodes at a given level (parsing, typechecking, static checking) before moving to the next level, rather than doing all checks for a given node before checking the next node"},
	{ "Show Error Windows Automatically", "autoShowErrors", "Automatically popup windows showing errors as items are checked" },
    };

    static final public Class guioptions = escjava.gui.GuiOptionsPanel.Settings.class;

    public GuiOptionsPanel() {
	init();
    }

    public void init() {
	setLayout(new BoxLayout(this,BoxLayout.PAGE_AXIS));
	removeAll();
	JCheckBox cb;
	for (int i = 0; i<info.length; ++i) {
	    String[] iinfo = info[i];
	    try {
		Field f = guioptions.getField(iinfo[1]);
		boolean b = f.getBoolean(settings);
		cb = new JCheckBox(iinfo[0],b);
		cb.setToolTipText(iinfo[2]);
		cb.addActionListener(this);
		add(cb);
	    } catch (Exception e) {
		System.out.println("FAILED TO RECOGNIZE OPTION " + i + ": " + iinfo[0] + " " + e);
	    }
	}
    }

    public void actionPerformed(/*@non_null*/ActionEvent e) {
	// write back out to the Options structure

	Object source = e.getSource();
	String name = null;
	if (source instanceof JCheckBox) {
	    String fname = null;
	    name = ((JCheckBox)source).getText();
	    for (int i=0; i<info.length; ++i) {
		if (info[i][0].equals(name)) {
		    fname = info[i][1];
		    break;
		}
	    }
	    boolean value = ((JCheckBox)source).isSelected();
	    try {
		Field f = guioptions.getField(fname);
		f.setBoolean(settings,value);
	    } catch (Exception ee) {
		System.out.println("FAILED TO RECOGNIZE OPTION: " + name + " " + ee);
	    }
	} else {
	    System.out.println("UNKNOWN GUI OPTION " + name);
	}
    }

}
