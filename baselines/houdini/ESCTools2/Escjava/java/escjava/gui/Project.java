package escjava.gui;

import javafe.InputEntry;
import junitutils.Utils;

import java.io.*;
import java.util.ArrayList;
import java.util.Iterator;
import java.lang.reflect.Field;

import javax.swing.JOptionPane;

public class Project {

    static String magic = "-EscProjectFileV0";  // See also escjava.Options
    static File currentProject = null;
    static final String eol = System.getProperty("line.separator");

    static public void init() {
        init(new String[0]);
    }

    static public void init(String[] args) {
	currentProject = null;
	GUI.gui.clear(false);
	ByteArrayOutputStream ba = Utils.setStreams();
	try {
	    GUI.gui.run(args);
	    GUI.gui.escframe.init();
	} finally {
	    Utils.restoreStreams(true);
	}
	String s = ba.toString();
	if (s.length() != 0) JOptionPane.showMessageDialog(
		GUI.gui.escframe,
		"Problems reported in reading the project file: " + eol +
		s);
    }


    static public void read(File file) {
	BufferedReader r = null;
	try {
	    r = new BufferedReader(new FileReader(file));
	    String s = r.readLine();
	    if (!s.equals(magic)) {
		JOptionPane.showMessageDialog(GUI.gui.escframe,
			"The selected file, " + 
			file.getAbsolutePath() + 
			", does not appear to be a project file");
		init();
		return;
	    }
	} catch (IOException e) {
		JOptionPane.showMessageDialog(GUI.gui.escframe,
			"The selected file, " + 
			file.getAbsolutePath() + 
			", could not be successfully read: " + e);
		init();
		return;
	} finally {
	    try {
		if (r != null) r.close();
	    } catch (IOException e) {
		JOptionPane.showMessageDialog(GUI.gui.escframe,
			"The selected file, " + 
			file.getAbsolutePath() + 
			", could not be successfully closed: " + e);
		init();
		return;
 	    }
	}
	// FIXME need to capture messages about invalid material in the
	// file.
	init(new String[]{ "-f", file.getAbsolutePath() });
	currentProject = file;
    }

    static public void write() {
	write(currentProject);
    }

    static public void write(File file) {
        FileWriter fw = null;
	try {
	    fw = new FileWriter(file);

	    // Write a header that marks this as a project file
	    fw.write(magic);
	    fw.write(eol);

	    // Write out all the options (that aren't defaults)
	    String s = optionString();
	    if (s != null) {
		fw.write(optionString());
		fw.write(eol);
	    }

	    // Write out the classpath
	    if (GUI.gui.options.userPath != null) {
		fw.write("-classpath ");
		fw.write(GUI.gui.options.userPath);
		fw.write(eol);
	    }
	    fw.write(eol);

	    // Write out the specs path
	    if (GUI.gui.options().specspath != null) {
		fw.write("-specs ");
		fw.write(GUI.gui.options().specspath);
		fw.write(eol);
	    }
	    fw.write(eol);

	    // Write out the input entries
	    Iterator i = GUI.gui.options.inputEntries.iterator();
	    while (i.hasNext()) {
		fw.write(((InputEntry)i.next()).savedString());
		fw.write(eol);
	    }

	    currentProject = file;
        } catch (Exception e) {
	} finally {
	    try {
		    if (fw != null) fw.close();
	    } catch (java.io.IOException e) {
	    }
	}
    }

	// FIXME - should really ahve this come from Options, rather
	// than being part of the GUI code
    static public String optionString() {
	String[][] optionarray = EscOptions.optionsShown;
	escjava.Options options = escjava.Main.options();
	StringBuffer sb = new StringBuffer(200);

	String s = System.getProperty("simplify");
	if (s != null && s.trim().length() != 0)
	    sb.append(" -simplify " + System.getProperty("simplify") + eol);
	for (int i=0; i<optionarray.length; ++i) {
	    String[] opt = optionarray[i];
	    if (opt[1]==null) continue;
	    try {
		Field f = EscOptions.escoptions.getField(opt[1]);
		boolean b = f.getBoolean(options);
		if (b) { sb.append(" -"); sb.append(opt[0]); }
	    } catch (Exception e) {
		System.out.println("FAILED TO RECOGNIZE FIELD " + opt[1]);
	    }
	}
	return sb.toString() + " " + GUI.gui.options().nowarnOptionString();
    }

}
