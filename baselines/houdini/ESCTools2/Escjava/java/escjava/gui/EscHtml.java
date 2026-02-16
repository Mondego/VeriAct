package escjava.gui;
import javax.swing.*;
import javax.swing.text.html.*;
import javax.swing.event.*;

import java.net.*;
import java.util.jar.*;

public class EscHtml extends JFrame {

    final JEditorPane editor;
    final JScrollPane scroll;
    private EscHtml() {
	editor = new JEditorPane();
	scroll = new JScrollPane(editor);
	editor.setContentType("text/html");
	editor.setEditable(false);
	editor.addHyperlinkListener(new Hyperactive());
	getContentPane().add(scroll);
    }

    /** Launches a non-editable HTML display window. */
    public static EscHtml make(String title, String filename, JFrame jf,
					int x, int y, int w, int h) {
	try {
	    java.net.URL url =
	      GUI.class.getClassLoader().getResource(filename);
	    return new EscHtml(title,url,x,y,w,h);
        } catch (Exception e) {
	    JOptionPane.showMessageDialog(jf,
                "Internal error - Could not find html file " + 
		filename + ":" + System.getProperty("line.separator")
                + e);
	}
	return null;
    }

    // Displays a local file that might be in the jar file
    public EscHtml(String title, java.net.URL url, int x, int y, int w, int h) 
					throws java.io.IOException {
	this();
	setTitle(title);
	editor.setPage(url);
	//pack();
	if (w != 0) setSize(w,h);
	setLocation(x,y);
    }

    public void showit() {
	FrameShower.show(this);
    }

    static class Hyperactive implements HyperlinkListener {

	public void hyperlinkUpdate(HyperlinkEvent e) {
	    if (e.getEventType() == HyperlinkEvent.EventType.ACTIVATED) {
		JEditorPane pane = (JEditorPane) e.getSource();
		if (e instanceof HTMLFrameHyperlinkEvent) {
		    HTMLFrameHyperlinkEvent  evt = (HTMLFrameHyperlinkEvent)e;
		    HTMLDocument doc = (HTMLDocument)pane.getDocument();
		    doc.processHTMLFrameHyperlinkEvent(evt);
		} else {
		    try {
			pane.setPage(e.getURL());
		    } catch (Throwable t) {
			t.printStackTrace();
		    }
		}
	    }
	}
    }
}
