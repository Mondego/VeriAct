package escjava.gui;
import java.awt.Dimension;
import java.awt.Rectangle;
import java.awt.Font;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;
import java.io.*;
import javax.swing.*;
import javax.swing.text.html.*;
import javax.swing.event.*;
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.util.Map;
import java.util.HashMap;
import java.net.*;
import java.util.jar.*;

public class EscEditor extends JFrame implements ActionListener {

    final JEditorPane editor;
    final JScrollPane scroll;
    final JComboBox fileChoice = null;
    String[] files;

    /** Launches an editable text window positioned at the given line,
	if line is positive (positioned at the beginning if line <= 0). 
<p>
	If istream is not null, then the data is read from that stream, and
	the filename is treated as the title.
<P>
	If istream is null, then the data is read from the filename.

    */
    public EscEditor(String filename, InputStream istream, int line, String[] files, String[] labels) {
	this.files = files;
	if (filename != null) setTitle(filename);


	// setJMenuBar(EscFrame.menubar); // It appears a JMenuBar
	// cannot be shared - FIXME - but want to use the menus while
	// looking at the documentation

	getContentPane().setLayout(new BoxLayout(getContentPane(),BoxLayout.PAGE_AXIS));
	final File f = istream != null ? null : new File(filename);
	JEditorPane editor = null;
	JScrollPane scroll = null;
	Reader r = null;
	try {
	    editor = new JEditorPane();
	    editor.setFont(new Font("Monospaced",Font.PLAIN,editor.getFont().getSize()));
	    scroll = new JScrollPane(editor);

	    JPanel jp = new JPanel();
	    jp.setLayout(new BoxLayout(jp,BoxLayout.LINE_AXIS));
	    getContentPane().add(jp);

	    JButton jb;
	    if (f != null) {
		jp.add(jb = new JButton("Save"));
		jb.addActionListener(new ActionListener() {
		    public void actionPerformed(/*@non_null*/ActionEvent e) {
			try {
			EscEditor.this.editor.write(new FileWriter(f));
			} catch (IOException ee) {} // FIXME
		    }
		});
		jp.add(jb = new JButton("Reload"));
		jb.addActionListener(new ActionListener() {
		    public void actionPerformed(/*@non_null*/ActionEvent e) {
			try {
			EscEditor.this.editor.read(new FileReader(f),null);
			} catch (IOException ee) {} // FIXME
		    }
		});
	    }
	    jp.add(new JLabel("Go to line:"));
	    final JTextField jt = new JTextField("");
	    jp.add(jt);
	    jt.setColumns(10);
	    jt.setMaximumSize(jt.getPreferredSize());
	    jt.addActionListener(new ActionListener() {
		public void actionPerformed(/*@non_null*/ActionEvent e) {
		    String s = jt.getText();
		    try {
			    int i = Integer.parseInt(s.trim());
			    EscEditor.this.scrollToLine(i);
		    } catch(Exception ee) {}
		}
	    });
	    if (files != null) {
		final JComboBox fileChoice = new JComboBox(labels);
		for (int i = 0; i<files.length; ++i) {
		    if (files[i].equals(filename)) fileChoice.setSelectedIndex(i);
		}
		jp.add(fileChoice);
		fileChoice.addActionListener(this);
	    }

	    getContentPane().add(scroll);
	    scroll.setPreferredSize(new Dimension(700,400));
	
	    editor.setEditable(f != null);
	    if (f != null) editor.addKeyListener(
		new KeyAdapter() {
		    public void keyTyped(KeyEvent e) {
			if (e.getModifiers() != KeyEvent.META_MASK) return;
			if (e.getKeyChar() != 's') return;
			try {
			    EscEditor.this.editor.write(new FileWriter(f));
			} catch (Exception ee) {
			    // FIXME - failed to save
		    }
		}
	    });
	
	    if (istream == null) {
		if (f != null) editor.setPage(f.toURL());
	    } else {
		// FIXME - It seems there ought to be a way to read from
		// the jar file and entry as a URL, but I cannot get that to
		// work
		
		char[] ca = new char[10000];
	        r = new InputStreamReader(
				new BufferedInputStream(istream));
		StringBuffer sb = new StringBuffer(10000);
		while (true) {  // FIXME - presuming ready() does not block
		    int n = r.read(ca);
		    if (n <= 0) break;
		    sb.append(ca,0,n);
		}
		editor.setText(sb.toString());

	    }

	    pack();
	    
	} catch (Exception e) {
	    if (editor != null) editor.setText("An exception occurred while trying to set up an editor for file " + filename + ": " + e);
	    line = 0;
	} finally {
	    try {
		if (r != null) r.close();
	    } catch (IOException e) {
		if (editor != null) editor.setText("An exception occurred while trying to set up an editor for file " + filename + ": " + e);
	    }
	}
	this.editor = editor;
	this.scroll = scroll;
	if (line>0 && editor != null) scrollToLine(line);
    }

    public void actionPerformed(/*@non_null*/ActionEvent e) {
	if (e.getSource() == fileChoice) {
	    int i = fileChoice.getSelectedIndex();
	    String filename = files[i];
	    try {
		editor.setPage(new File(filename).toURL());
		setTitle(filename);
	    } catch (Exception ee) {
		JOptionPane.showMessageDialog(this,
		    "Failed to read from file " + filename + ": " +
		    ee);
	    }
	}
    }

    public void scrollToLine(int line) {
	try { if (line > 0) { 
	    int ceol = Project.eol.charAt(0);
	    int neol = Project.eol.length();
	    String text = editor.getText();
	    int i = line;
	    int npos = -neol;
	    int pos = 0;
	    while (--i>=0) {
		pos = npos + neol;
		npos = text.indexOf(ceol,pos);
		if (npos == -1) {
		    npos = text.length()-1;
		    break;
		}
	    }
	    Rectangle r = editor.modelToView(pos);
	    int height = scroll.getViewport().getExtentSize().height;
	    int top = r.y - height*4/10;
	    r = new Rectangle(0,top,50,top+height*9/10);
	    editor.scrollRectToVisible(r);
	    //editor.setCaretPosition(pos);
	    //editor.moveCaretPosition(npos-1);
	    editor.select(pos,npos-1);
//	    editor.setSelectionStart(pos);
//	    editor.setSelectionEnd(npos-1);
	    pack();
	}} catch (Exception e) {} 
		// If an exception occurs, just don't bother to scroll
    }


    static Map map = new HashMap();

    static public EscEditor make(String filename, int line, 
				String[] files, String[] labels) {
	try {

	    int p = filename.indexOf(".jar:");
	    File f = null;
	    String id;
	    InputStream is = null;
	    if (p != -1) {
		String jarname = filename.substring(0,p+4);
		String fn = filename.substring(p+5);
		JarFile jf = new JarFile(jarname);
		JarEntry je = jf.getJarEntry(fn);
		id = ("jar:file:/" + jarname + "!/" + fn);
		is = jf.getInputStream(je);
		
	    } else {
		f = new File(filename);
		id = f.getCanonicalPath();
	    }
	    Object o = map.get(id);
	    EscEditor e;
	    if (o == null) {
		    e = new EscEditor(filename,is,line,files,labels);
		    map.put(id,e);
	    } else {
		    e = (EscEditor)o;
		    e.scrollToLine(line);
	    }
	    return e;

	} catch (Exception ee) {
	    EscEditor ed = new EscEditor(null,null,1,null,null);
	    ed.editor.setText("ERROR - Failed to create an editor for file " + filename + " :: " + ee);
	    return ed;
	}
    }

    public void showit() {
	FrameShower.show(this);
    }
}
