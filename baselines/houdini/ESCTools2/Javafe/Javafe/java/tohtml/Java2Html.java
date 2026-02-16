/*  Java2HTML Converter

Converts all the .java files in an input directory (and all
sub-directories) into browsable HTML. An index HTML file lists
all the packages, from where you can navigate to the source code
of the classes belonging to that file. Java keywords and comments
are nicely highlighted in the HTML'ized Java code.

The program takes two arguments (sorry, no error checking):

    - Name of the input directory to convert
    - Name of the output directory where HTML is output
    
Hannes Marais,
marais@pa.dec.com
June 1998

Extended by Cormac Flanagan and Rustan Leino, Sep 1999.

Copyright 1998, 1999, Compaq Computer Corporation.

This software is a
research work and is provided "as is." Compaq disclaims all express
and implied warranties with regard to such software, including the
warranty of fitness for a particular purpose.  In no event shall
Compaq be liable for any special, direct, indirect, or consequential
damages or any other damages whatsoever, including any loss or any
claim for loss of data or profits, arising out of the use or inability
to use the software even if Compaq has been advised of the possibility
of such damages. This software shall not be further distributed
without prior written permission from Compaq Computer Corporation.

*/

package tohtml;

import java.io.*;
import java.util.*;

public class Java2Html 
{
    static public String destination;
    static /*@ non_null */Vector packages = new Vector();
    //@ invariant packages.elementType == \type(Package);
    //@ invariant !packages.containsNull;
    static long lines = 0;

    static char annotationChar = '@';

    static final String markupDeclLinkBegin = "<font color=000000>";
    static final String markupDeclLinkEnd = "</font>";
    static final String markupKeywordBegin = "<font color=0000aa><b>";
    static final String markupKeywordEnd = "</b></font>";
    static final String markupCommentBegin = "<font color=00aa00>";
    static final String markupCommentEnd = "</font>";
    static final String markupPragmaBegin = "<font color=00aa00><b>";
    static final String markupPragmaEnd = "</b></font>";
    static final String markupWizardPragmaBegin = "<font color=aa0022><b>";
    static final String markupWizardPragmaEnd = "</b></font>";
    static final String markupRemovedPragmaBegin = "<font color=aaaaaa>";
    static final String markupRemovedPragmaEnd = "</font>";
    static final String markupLineNumberBegin = "<font size=-1 color=000000>";
    static final String markupLineNumberEnd = "</font>";
  
    public static void ReadDir(Package parent,
			       /*@ non_null */ String name,
			       /*@ non_null */ String srcdir) {
        Package P = new Package(parent, name, srcdir);
        packages.addElement(P);
        
        File F = new File(srcdir);
        String file[] = F.list();
	/*@ assume \nonnullelements(file); */  // we assume srcdir denotes a dir.
        for(int i = 0; i < file.length; i++) {
            String f = file[i];
		// FIXME -- will want to expand (or make extensible) this list of suffixes
            if (f.endsWith(".java") || f.endsWith(".spec") ) 
                P.AddFile(f);
            else {
                String dirname = srcdir + File.separator + f;
                File D = new File(dirname);
                if (D.isDirectory()) {
                    if (name.equals(""))
                        ReadDir(P, f, dirname);
                    else
                        ReadDir(P, name + "." + f, dirname);
                }
            }
        }
    }
    
    public static void WritePackageList() {
        String fname = Java2Html.destination + File.separator + "packages.html";
            
        try {
            PrintWriter W = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(fname))));
            W.println("<html><head><title>All packages</title></head><body bgcolor=ffffdd>");
            W.println("<b>All packages</b><hr>");
                    
            Enumeration e = packages.elements();
            while(e.hasMoreElements()) {
                Package p = (Package)e.nextElement();
                File F = new File(Java2Html.destination + File.separator + p.IndexName());
                if (F.exists()) {
                    if (p.name.equals("")) 
                        W.println("<A HREF=" + p.IndexName() + "><b>default package</b></A></BR>");
                    else
                        W.println("<A HREF=" + p.IndexName() + "><b>" + p.name + "</b></A></BR>");
                }
            }    
            W.println("</body></html>");
            W.close();
        } catch (IOException e) {
	    System.out.println("IOException: " + e);
        }
    }

    // Maps filename to hashtable mapping offsets (Long) to String "<A HREF=...">
    static /*@ non_null */ Hashtable declLinks = new Hashtable();
    //@ invariant declLinks.elementType == \type(Hashtable);
    //@ invariant declLinks.keyType == \type(String);


    //@ requires \nonnullelements(args);
    //@ requires \nonnullelements(args);
    public static void main(String[] args) {

	if (args.length<3) {
	    System.out.println("Usage: html-dir dest annotatin-char");
	    System.exit(1);
	}

	if( args.length>3 ) {
	    readDeclLinks(args[3]);
	}
        
	annotationChar = args[2].charAt(0);

        destination = args[1];
        ReadDir(null, "", args[0]);

        VectorSorter.Sort(packages);
        
        WritePackageList();
        Enumeration e = packages.elements();
        while(e.hasMoreElements()) {
            Package P = (Package)e.nextElement();
            P.Convert2Html();
        }
        System.out.println("Lines of code: " + lines);
    }

    private static void readDeclLinks(/*@ non_null */ String filename) {
	try {
	    StringBuffer buf = new StringBuffer();
	    Reader R = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
	    
	    while(true) {
		int ch = R.read();
		buf.setLength(0);
		
		while (ch != -1 && ch != '\n') {
		    buf.append((char)ch);
		    ch = R.read();
		}
		
		if( ch == -1 ) break;
		
		String s = buf.toString();
		int i1=s.indexOf(' ');
		int i2=s.indexOf(' ',i1+1);
		int i3=s.indexOf(' ',i2+1);
		// System.out.println("Link <"+s+">"+" "+i1+" "+i2+" "+i3);
		String filename1 = s.substring(0,i1);
		Long offset = new Long(s.substring(i1+1,i2));
		String filename2 = s.substring(i2+1,i3);
		Long linenumber = new Long(s.substring(i3+1));
		String href = "<A HREF=\"" + JFile.filenameFlatten(filename2) + ".html#" + linenumber + "\">";
		// System.out.println("<"+filename1+"> "+offset+" "+href);
		//@ assume filename1 != null;
		Hashtable h = (Hashtable)declLinks.get(filename1);
		if( h==null ) {
		    h = new Hashtable();
		    declLinks.put(filename1,h);
		}
		//@ assume h.elementType == \type(String);
		//@ assume h.keyType == \type(Long);
		h.put(offset,href);
	    }
        } catch(IOException e) {
	    System.out.println("IOException: " + e);
        } catch (Exception e) {
	    System.out.println("Exception: " + e);
	}
    }	    
	
	
}

class Package {
    String sourcedir;
    /*@ non_null */ String name;
    Package parent;
    
    /*@ non_null*/ Vector files = new Vector();
    //@ invariant files.elementType == \type(JFile);
    //@ invariant !files.containsNull;

    // It seems that "subpackages" is never used.  -- KRML
    /*@ non_null */ Vector subpackages = new Vector();
    //@ invariant subpackages.elementType == \type(Package);
    //@ invariant !subpackages.containsNull;

    //@ invariant files.owner == this;
    //@ invariant subpackages.owner == this;
    //@ invariant files != subpackages;

    public Package(Package parent, /*@ non_null */ String name,
		   String sourcedir) {
        //@ set subpackages.elementType = \type(Package);
        //@ set subpackages.containsNull = false;
        //@ set subpackages.owner = this;
        //@ set files.elementType = \type(JFile);
        //@ set files.containsNull = false;
        //@ set files.owner = this;
        this.name = name;
        this.sourcedir = sourcedir;
        this.parent = parent;
        if (parent != null)
            parent.subpackages.addElement(this);
    }
    
    public void AddFile(/*@ non_null */ String name) {
        files.addElement(new JFile(name, this));
    }
    
    //@ ensures \result != null;
    public String FullClassName(/*@ non_null */ JFile f) {
        if (name.equals(""))
            return f.name;
        else
            return name + "." + f.name;
    }

    //@ ensures \result != null;
    public String SourceFilename(/*@ non_null */ JFile f) {
        return sourcedir + File.separator + f.name;
    }
    
    public String IndexName() {
        if (name.equals(""))
            return "default-package-idx.html";
        else
            return name + "-idx.html";
    }
        
    public void Convert2Html() {
        if (files.size() > 0) {
            VectorSorter.Sort(files);
            
            // convert all the files of this package
            Enumeration e = files.elements();
            while(e.hasMoreElements()) {
                JFile f = (JFile)e.nextElement();
                f.Convert2Html();
            }
                    
            // write an index for this package
            String indexname = Java2Html.destination + File.separator + IndexName();
            try {
                PrintWriter W = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(indexname))));
                    
                if (name.equals("")) {
                    W.println("<html><head><title>default package</title></head><body>");
                    W.println("<b>default package</b><hr>");
                } else {
                    W.println("<html><head><title>package " + name + "</title></head><body bgcolor=ffffdd>");
                    W.println("<b>package " + name + "</b><hr>");
                }
                
                e = files.elements();
                while(e.hasMoreElements()) {
                    JFile f = (JFile)e.nextElement();
                    W.println("<A HREF=" + f.FullClassName() + ".html>" + f.name + "</A></BR>");
                }
                
                W.println("</body></html>");
                W.close();
            } catch (IOException ex) {
		System.out.println("IOException: " + ex);
            }
	}
    }    

}

class JFile {
    /*@ non_null */ Package pack;
    /*@ non_null */ String name;            // filename
    /*@ non_null */ Hashtable declLinks;
    //@ invariant declLinks.elementType == \type(String);
    //@ invariant declLinks.keyType == \type(Long);

    public JFile(/*@ non_null */ String name, 
		 /*@ non_null */ Package pack) {
        this.name = name;
        this.pack = pack;
	//@ set keywords.keyType = \type(String);
	declLinks = (Hashtable)Java2Html.declLinks.get( SourceFilename() ); //@ nowarn;
	//@ assume declLinks.elementType == \type(String);
	//@ assume declLinks.keyType == \type(Long);
	// System.out.println("JFile: "+SourceFilename() );
	if( declLinks == null ) {
	    declLinks = new Hashtable();
	    System.out.println("JFile with no links: "+SourceFilename() );
	}
    }
    
    //@ ensures \result != null;
    public String SourceFilename() {
        return pack.SourceFilename(this);
    }
    
    //@ ensures \result != null;
    public String FullClassName() {
        return pack.FullClassName(this);
    }
    
    final /*@ non_null */ StringBuffer buf = new StringBuffer();
    
    final void Flush(/*@ non_null */ LineNumberPrintWriter W) {
        if (buf.length() > 0) {
            String word = buf.toString();
            boolean key = keywords.get(word) != null;
	    String href = (String)declLinks.get(new Long(offset-buf.length()));
            
	    if( href != null ) W.print(href+Java2Html.markupDeclLinkBegin);
            if (key) W.print(Java2Html.markupKeywordBegin);
            W.print(word);
            if (key) W.print(Java2Html.markupKeywordEnd);
	    if( href != null ) W.print("</A>"+Java2Html.markupDeclLinkEnd);
	    buf.setLength(0);
        }
    }
    
    public void printCode(/*@ non_null */ LineNumberPrintWriter W,
			  /*@ non_null */ String s) {
	int n = s.length();
	for (int i = 0; i < n; i++) {
	    printCode(W, s.charAt(i));
	}
    }

    public void printCode(/*@ non_null */ LineNumberPrintWriter W, char ch) {
        if (ch == '<')
            W.print("&lt;");
        else if (ch == '>')
            W.print("&gt;");
        else if (ch == '&')
            W.print("&amp;");
        else if (ch == '\r')
		;
        else
            W.print(ch);
    }

    /*@ spec_public */ private Reader R;
    private int offset=0;

    //@ requires R != null;
    int read() throws java.io.IOException {
	offset++;
	return R.read();
    }


    public void Convert2Html() {
        try {
            System.out.println(FullClassName());
            String srcname = SourceFilename();
            String dstname = Java2Html.destination + File.separator + FullClassName() + ".html";
            R = new BufferedReader(new InputStreamReader(new FileInputStream(srcname)));
            PrintWriter pw = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(dstname))));
	    LineNumberPrintWriter W = new LineNumberPrintWriter(pw);
            
	    //            W.println("<html><head><title>Source of " + FullClassName() + "</title></head>");
            W.println("<html><head><title>" + name +" in package "+pack.name+"</title></head>");
            W.println("<body bgcolor=\"#ffffdd\" link=\"#aaaaaa\" vlink=\"#aaaaaa\" alink=\"#aaaaaa\">");
            W.println("<b>" + FullClassName() + "</b>");
            W.println("<HR><pre>");
	    W.enterCodeSegment();
            int ch = read();
            while (ch != -1) {
                if (Alpha(ch)) {
                    buf.append((char)ch);
                } else {
                    Flush(W);
                    if (ch == '"' || ch == '\'') {     // skip strings
                        int fin = ch;
                        printCode(W, (char)ch);
                        ch = read();
                        while (ch != -1 && ch != fin) {
                            if (ch == '\\') {            // escape, takes care of \' and \"
                                printCode(W, (char)ch);
                                ch = read();
                            }
                            printCode(W, (char)ch);
                            ch = read();
                        }
                        printCode(W, (char)ch);
                    } else if (ch == '/') {
                        ch = read();
                        
                        if (ch == '/') {
			    ch = read();
			    boolean isPragma = ch == '@';
			    if (isPragma) {
				W.print(Java2Html.markupPragmaBegin);
			    } else {
				W.print(Java2Html.markupCommentBegin);
			    }
			    W.print("//");
			    while (ch != -1 && ch != '\n') {
				printCode(W, (char)ch);
				ch = read();
			    }
			    if (isPragma) {
				W.print(Java2Html.markupPragmaEnd);
			    } else {
				W.print(Java2Html.markupCommentEnd);
			    }
			    printCode(W, '\n');

                        } else if (ch == '*') {
			    String s = "/*" + readUntilEndOfComment();

			    if (s.startsWith("/*" + Java2Html.annotationChar + "(")) {
				W.print(Java2Html.markupWizardPragmaBegin);
				int k = s.indexOf(')');
				if (k != -1) {
				    s = "/*" + Java2Html.annotationChar + 
					s.substring(k+1);  // skip to beyond the ')'
				}
				printCode(W, s);
				W.print(Java2Html.markupWizardPragmaEnd);
			    
			    } else if (s.startsWith("/*" + Java2Html.annotationChar)) {
				W.print(Java2Html.markupPragmaBegin);
				printCode(W, s);
				W.print(Java2Html.markupPragmaEnd);
			    
			    } else if (s.startsWith("/* REMOVED " + 
						    Java2Html.annotationChar + "(")) {
				String pre = Java2Html.markupRemovedPragmaBegin;
				String post = Java2Html.markupRemovedPragmaEnd;

				int k = s.indexOf(')');
				if (k != -1) {
				    String t = s.substring(k+1);  // skip to beyond the ')'
				    k = t.indexOf(" BECAUSE ");
				    if (k != -1) {
					s = "/*" + t.substring(0, k) + " */";
					t = t.substring(k + " BECAUSE ".length());

				// try to add hyperlink
					k = t.indexOf(".java:");
				// if (k == -1) k = t.indexOf(".spec:");
					if (k != -1) {
					    String file = filenameFlatten(t.substring(0, k)) +
				                ".java.html";
					    t = t.substring(k + ".java:".length());
					    k = t.indexOf(':');
					    if (k != -1) {
						String line = t.substring(0, k);
						pre = "<A HREF=\"" + file + "#" + line + "\">";
						post = "</A>";
					    }
					}
				    }
				}
				W.print(pre);
				printCode(W, s);
				W.print(post);
			    
			    } else {
				W.print(Java2Html.markupCommentBegin);
				printCode(W, s);
				W.print(Java2Html.markupCommentEnd);
			    }
                        } else {
                            printCode(W, '/');
                            printCode(W, (char)ch);
                        }
                    } else {
                        if (ch == '\n') Java2Html.lines++;
                        printCode(W, (char)ch);
                    }
                }
                ch = read();
            }
            Flush(W);
	    W.exitCodeSegment();
	    /* Add blank lines so hyperlinks to internal tags 
	       go to right place near end of file */
	    for(int i=0; i<100; i++) W.println("\n");
            W.println("</pre></body></html>");
            W.close();
        } catch(IOException e) {
	    System.out.println("IOException: " + e);
	    System.exit(1);
        }
    }

    /** Removes the "./" from the beginning of <code>name</code> (if
     * present) and changes all remaining occurrences of "/" into "."
     * (where in both cases with "/" we actually mean File.separator).
     **/
    
    static String filenameFlatten(/*@ non_null */ String name) {
	if (name.startsWith("." + File.separator)) {
	    name = name.substring(2);
	}
	while (true) {
	    int k = name.indexOf(File.separator);
	    if (k == -1) {
		break;
	    }
	    name = name.substring(0, k) + "." + name.substring(k+1);
	}
	return name;
    }
  
    //@ requires R != null;
    private String readUntilEndOfComment()
	throws java.io.IOException {
	StringBuffer sb = new StringBuffer();
    
	int cha = read();
	while (cha != -1) {
	    int chb = read();
	    if (cha == '*' && chb == '/') {
		sb.append("*/");
		break;
	    }
	    sb.append((char)cha);
	    cha = chb;
	}
	return sb.toString();
    }

    boolean Alpha(int ch) {
	//        return ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z' || ch >= '0' && ch <= '9';
	return Character.isJavaIdentifierPart((char)ch);
    }
    
    static /*@ non_null */ Hashtable keywords = new Hashtable();
    //@ invariant keywords.keyType == \type(String);
    
    static {
        String kw = "boolean|char|byte|short|int|long|float|double|abstract|break|byvalue|case|cast|catch|"+
            "class|const|continue|default|do|else|extends|"+
            "false|final|finally|for|future|generic|goto|if|"+
            "implements|import|inner|instanceof|interface|"+
            "native|new|null|operator|outer|package|private|"+
            "protected|public|rest|return|static|super|switch|"+
            "synchronized|this|throw|throws|transient|true|try|"+
            "var|volatile|while|Boolean|Byte|Character|Class|ClassLoader|Cloneable|Compiler|"+
            "Double|Float|Integer|Long|Math|Number|Object|Process|"+
            "Runnable|Runtime|SecurityManager|Short|String|StringBuffer|"+
            "System|Thread|ThreadGroup|Void";
            
        Object x = new Object();
        StringTokenizer st = new StringTokenizer(kw, "|");
        while (st.hasMoreTokens()) {
            keywords.put(st.nextToken(), x);
        }
    }    
}

class VectorSorter {
    static private /*@non_null*/ Vector dat = new Vector();

    /*@ requires data.elementType == \type(Package) ||
                 data.elementType == \type(JFile); */
    //@ requires !data.containsNull;
    static public void Sort(/*@ non_null */ Vector data) {
        dat = data;
        sort(0, dat.size() - 1);
    }

    /*@ requires (a instanceof Package && b instanceof Package) ||
                 (a instanceof JFile && b instanceof JFile); */
    static private final long compare(Object a, Object b) {
        if (a instanceof Package)
            return ((Package)a).name.compareTo(((Package)b).name);
        else
            return ((JFile)a).name.compareTo(((JFile)b).name);
    }

    //@ requires 0 <= p;
    //@ requires r < dat.elementCount;

    /*@ requires dat.elementType == \type(Package) ||
                 dat.elementType == \type(JFile); */
    //@ requires !dat.containsNull;
    static private void sort(int p, int r) {
        if (p < r) {
            int q = partition(p,r);
            if (q == r) 
                q--;
            sort(p,q);
            sort(q+1,r);
        }
    }

    //@ requires 0 <= lo;
    //@ requires lo <= hi;
    //@ requires hi < dat.elementCount;

    /*@ requires dat.elementType == \type(Package) ||
                 dat.elementType == \type(JFile); */
    //@ requires !dat.containsNull;
    //@ ensures lo <= \result && \result <= hi;
    static private int partition(int lo, int hi) {
        Object pivot = dat.elementAt(lo);
        while (true) {
            while (compare(dat.elementAt(hi), pivot) >= 0 && lo < hi) hi--;
            while (compare(dat.elementAt(lo), pivot) < 0 && lo < hi) lo++;
            if (lo < hi) {
                Object T = dat.elementAt(lo);
                dat.setElementAt(dat.elementAt(hi), lo);
                dat.setElementAt(T, hi);
            } else
                return hi;
        }
    }
}

class LineNumberPrintWriter {
    /*@ spec_public */ private boolean inCodeSegment = false;
    private /*@ non_null */ PrintWriter pw;
    private long lineNumber = 1;
    private boolean atBeginningOfLine = true;

  //@ ensures !inCodeSegment;
    LineNumberPrintWriter(/*@ non_null */ PrintWriter pw) {
	this.pw = pw;
    }

    //@ requires !inCodeSegment;
    //@ modifies inCodeSegment;
    //@ ensures inCodeSegment;
    public void enterCodeSegment() {
	inCodeSegment = true;
	atBeginningOfLine = true;
    }
  
    //@ requires inCodeSegment;
    //@ modifies inCodeSegment;
    //@ ensures !inCodeSegment;
    public void exitCodeSegment() {
	if (!atBeginningOfLine) {
	    lineNumber++;
	    atBeginningOfLine = true;
	}
	inCodeSegment = false;
    }

    public void print(/*@ non_null */ String s) {
	int n = s.length();
	for (int i = 0; i < n; i++) {
	    print(s.charAt(i));
	}
    }

    public void print(char ch) {
	if (inCodeSegment) {
	    if (atBeginningOfLine) {
		String s = "     " + lineNumber;
		pw.print("<A NAME=\"" + lineNumber + "\"></A>");
		pw.print(Java2Html.markupLineNumberBegin);
		pw.print(s.substring(s.length()-5) + "  ");
		pw.print(Java2Html.markupLineNumberEnd);
		atBeginningOfLine = false;
	    }
	}
	if( ch == '\t') {
	    pw.print("        ");
	} else {
	    pw.print(ch);
	}
	if (inCodeSegment && ch == '\n') {
	    atBeginningOfLine = true;
	    lineNumber++;
	}
    }

    public void println(/*@ non_null */ String s) {
	print(s);
	print('\n');
    }

    public void close() {
	pw.close();
    }
}
