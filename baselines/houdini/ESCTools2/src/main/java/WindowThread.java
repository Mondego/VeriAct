package escjava.gui;

import javax.swing.*;

public class WindowThread extends Thread {

    static TaskQueue windowTasks = new TaskQueue();

    public void run() {
	while (true) {
	    Object o = windowTasks.getTask();
	    String out = "";
	    if (o instanceof GUI.EscTreeValue) {
		((GUI.EscTreeValue)o).showOutput(true);
	    } else if (o instanceof HtmlTask) {
		HtmlTask ht = (HtmlTask)o;
		EscHtml jf = EscHtml.make(ht.title,ht.filename,
		    GUI.gui.escframe,500,0,600,400);
		jf.showit();
	    } else if (o instanceof String) {
		// Pop up an editor window
		try {
		    // Here we parse the various kinds of error messages
		    // that Esc/java2 produces to find a file name and
		    // line number
		    final String aheader =
			"Associated declaration is \"";
		    final String linetext = "\", line ";
		    final String coltext = ", col ";
		    String s = (String)o;
		    String name;
		    int lin;
		    if (s.startsWith(aheader)) {
			int i = s.indexOf('\"',aheader.length());
			name = s.substring(aheader.length(),i);
			i += linetext.length();
			int j = s.indexOf(',',i);
			try {
			    lin = Integer.parseInt(s.substring(i,j));
			} catch (NumberFormatException e) {
			    lin = 0;
			}
		    } else {
			final String jarending = ".jar:";
			int i = s.indexOf(jarending);
			i = i == -1 ? 0 : i + jarending.length();
			i = s.indexOf(':',i);
			if (i == -1) continue;
			name = s.substring(0,i);
			i++;
			int ii = s.indexOf(':',i);
			if (ii == -1) continue;
			try {
			    lin = Integer.parseInt(s.substring(i,ii));
			} catch (NumberFormatException e) {
			    lin = 0;
			}
		    }
		    if (name.endsWith(".class")) {
			int result =
			JOptionPane.showConfirmDialog(GUI.gui.escframe,
			    "The referenced location is in a class file, "
			  + " so there is probably no java or specification"
			  + Project.eol
			  + " file for this class.  Would you like to "
			  + " create a skeleton specification file and edit it?" + Project.eol + "[[[ Sorry - not yet implemented ]]]",
			    "Generate skeleton?",
			    JOptionPane.YES_NO_OPTION);
			if (result == JOptionPane.NO_OPTION) continue;
			continue;

		    }
		    EscEditor.make(name,lin,null,null).showit();
		} catch (Exception e) {
		    JOptionPane.showMessageDialog(GUI.gui.escframe,
			"Failed to open editor window: " + Project.eol +
			out +
			o + Project.eol + e);
		}
	    }
	}
    }

    static class HtmlTask {
        String filename;
        String title;
        public HtmlTask(String title, String filename) {
            this.title = title;
            this.filename = filename;
        }
    }

}

