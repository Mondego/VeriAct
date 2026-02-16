/*  Copyright 2004, David R. Cok 
    Originally generated as part of a GUI interface to the
    Esc/Java2 application.
*/

package escjava.gui;

import javax.swing.*;
import javax.swing.event.*;
import java.awt.*;
import java.awt.event.*;

/** This class generates a JFrame that contains the given
    uneditable text.  The text is in a Monospaced font
    appropriate for showing text where column alignment from
    line to line is important.
 */
public class EscOutputFrame extends JFrame {

    /** Create a JFrame with the given frame title and text.
     */
    //@ requires title != null;
    //@ requires text != null;
    public EscOutputFrame(String title, String text) {
	textArea = new JTextArea(text);
	textArea.setFont(new Font("Monospaced",Font.PLAIN,textArea.getFont().getSize()));
	textArea.setColumns(50);
	textArea.setEditable(false);
	textArea.setLineWrap(true);
	textArea.addMouseListener( new MouseInputAdapter() {
	    /* On receiving a mouse click, we identify the line in which
	       the click occurred, grab the text for that line and send the
	       text for that line as a task to the windowTasks queue (which
	       is supposed to launch an editor).
	    */
	    public void mouseClicked(/*@non_null*/ MouseEvent e) {
		JTextArea textArea = (JTextArea)e.getSource();
		try {
		    int p = textArea.getCaretPosition();
		    int line = textArea.getLineOfOffset(p);
		    textArea.setSelectionStart(textArea.getLineStartOffset(line));
		    textArea.setSelectionEnd(textArea.getLineEndOffset(line));
		    String s = textArea.getSelectedText();
		    WindowThread.windowTasks.addTask(s);
		} catch (Exception ee) {}
			// If an exception occurs, we simply don't display a window
	    }
	});
	JScrollPane scroll = new JScrollPane(textArea);
	scroll.setPreferredSize(new Dimension(400,300));
	getContentPane().add(scroll, BorderLayout.CENTER);
	setLocation(500,100);
	setTitle(title);
	pack();
	showit();
    }

    /** Causes the event subsystem to display the frame. */
    public void showit() {
	FrameShower.show(this);
    }

    private /*@ non_null */ JTextArea textArea;

    /** Appends text to that already contained in the frame. */
    //@ requires s != null;
    public void appendText(String s) {
	textArea.append(s);
	showit();
    }

    /** Replaces the text in the frame with the given text. */
    //@ requires s != null;
    public void setText(String s) {
	textArea.setText(s);
	showit();
    }
}
