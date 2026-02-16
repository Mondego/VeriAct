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

Copyright 1999, Compaq.

Copyright 1998 Digital Equipment Corporation. This software is a
research work and is provided "as is." Digital disclaims all express
and implied warranties with regard to such software, including the
warranty of fitness for a particular purpose.  In no event shall
Digital be liable for any special, direct, indirect, or consequential
damages or any other damages whatsoever, including any loss or any
claim for loss of data or profits, arising out of the use or inability
to use the software even if Digital has been advised of the possibility
of such damages. This software shall not be further distributed
without prior written permission from Digital Equipment Corporation.

*/

package tohtml;

import java.io.*;
import java.util.*;

public class Java2Html 
{
  static public String destination;
  static Vector packages = new Vector();
  static long lines = 0;

  static final String markupKeywordBegin = "<font color=0000aa><b>";
  static final String markupKeywordEnd = "</b></font>";
  static final String markupCommentBegin = "<font color=00aa00>";
  static final String markupCommentEnd = "</font>";
  static final String markupPragmaBegin = "<font color=00aa00><b>";
  static final String markupPragmaEnd = "</b></font>";
  static final String markupWizardPragmaBegin = "<font color=aa0022><b>";
  static final String markupWizardPragmaEnd = "</b></font>";
  static final String markupRemovedPragmaBegin = "<font color=aaaaaa";
  static final String markupRemovedPragmaEnd = "</font>";
  static final String markupLineNumberBegin = "<font size=-2 color=000000>";
  static final String markupLineNumberEnd = "</font>";
  
    public static void ReadDir(Package parent,
			       String name,
			       String srcdir) {
        Package P = new Package(parent, name, srcdir);
        packages.addElement(P);
        
        File F = new File(srcdir);
        String file[] = F.list();
        for(int i = 0; i < file.length; i++) {
            String f = file[i];
            if (f.endsWith(".java")) 
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

    public static void main(String[] args) {
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
}

class Package {
    String sourcedir;
    String name;
    Package parent;
    
    Vector files = new Vector();

    // It seems that "subpackages" is never used.  -- KRML
    Vector subpackages = new Vector();

    public Package(Package parent, String name,
		   String sourcedir) {
        this.name = name;
        this.sourcedir = sourcedir;
        this.parent = parent;
        if (parent != null)
            parent.subpackages.addElement(this);
    }
    
    public void AddFile(String name) {
        files.addElement(new JFile(name, this));
    }
    
    public String FullClassName(JFile f) {
        if (name.equals(""))
            return f.name;
        else
            return name + "." + f.name;
    }

    public String SourceFilename(JFile f) {
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
    Package pack;
    String name;            // filename

    public JFile(String name, Package pack) {
        this.name = name;
        this.pack = pack;
    }
    
    public String SourceFilename() {
        return pack.SourceFilename(this);
    }
    
    public String FullClassName() {
        return pack.FullClassName(this);
    }
    
    char[] buf = new char[1024];
    int pos = 0;
    
    final void Flush(LineNumberPrintWriter W) {
        if (pos > 0) {
            String word = new String(buf, 0, pos);
            boolean key = keywords.get(word) != null;
            
            if (key) W.print(Java2Html.markupKeywordBegin);
            W.print(word);
            if (key) W.print(Java2Html.markupKeywordEnd);
            pos = 0;
        }
    }
    
    public void printCode(LineNumberPrintWriter W,
			  String s) {
      int n = s.length();
      for (int i = 0; i < n; i++) {
	printCode(W, s.charAt(i));
      }
    }

    public void printCode(LineNumberPrintWriter W, char ch) {
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

    public void Convert2Html() {
        try {
            System.out.println(FullClassName());
            String srcname = SourceFilename();
            String dstname = Java2Html.destination + File.separator + FullClassName() + ".html";
            Reader R = new BufferedReader(new InputStreamReader(new FileInputStream(srcname)));
            PrintWriter pw = new PrintWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(dstname))));
	    LineNumberPrintWriter W = new LineNumberPrintWriter(pw);
            
            W.println("<html><head><title>Source of " + FullClassName() + "</title></head><body bgcolor=\"#ffffdd\" link=\"#aaaaaa\" vlink=\"#aaaaaa\" alink=\"#aaaaaa\">");
            W.println("<b>" + FullClassName() + "</b>");
            W.println("<HR><pre>");
	    W.enterCodeSegment();
            int ch = R.read();
            while (ch != -1) {
                if (Alpha(ch)) {
                    buf[pos++] = (char)ch;
                } else {
                    Flush(W);
                    if (ch == '"' || ch == '\'') {     // skip strings
                        int fin = ch;
                        printCode(W, (char)ch);
                        ch = R.read();
                        while (ch != -1 && ch != fin) {
                            if (ch == '\\') {            // escape, takes care of \' and \"
                                printCode(W, (char)ch);
                                ch = R.read();
                            }
                            printCode(W, (char)ch);
                            ch = R.read();
                        }
                        printCode(W, (char)ch);
                    } else if (ch == '/') {
                        ch = R.read();
                        
                        if (ch == '/') {
			  ch = R.read();
			  boolean isPragma = ch == '@';
			  if (isPragma) {
			    W.print(Java2Html.markupPragmaBegin);
			  } else {
                            W.print(Java2Html.markupCommentBegin);
			  }
			  W.print("//");
			  while (ch != -1 && ch != '\n') {
			    printCode(W, (char)ch);
			    ch = R.read();
			  }
			  if (isPragma) {
			    W.print(Java2Html.markupPragmaEnd);
			  } else {
                            W.print(Java2Html.markupCommentEnd);
			  }
			  printCode(W, '\n');

                        } else if (ch == '*') {
			  String s = "/*" + readUntilEndOfComment(R);

			  if (s.startsWith("/*@(")) {
			    W.print(Java2Html.markupWizardPragmaBegin);
			    int k = s.indexOf(')');
			    if (k != -1) {
			      s = "/*@" + s.substring(k+1);  // skip to beyond the ')'
			    }
			    printCode(W, s);
			    W.print(Java2Html.markupWizardPragmaEnd);
			    
			  } else if (s.startsWith("/* REMOVED @(")) {
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
                ch = R.read();
            }
            Flush(W);
	    W.exitCodeSegment();
            W.println("</pre></body></html>");
            W.close();
        } catch(IOException e) {
	  System.out.println("IOException: " + e);
        }
    }

  /** Removes the "./" from the beginning of <code>name</code> (if
    * present) and changes all remaining occurrences of "/" into "."
    * (where in both cases with "/" we actually mean File.separator).
    **/
  
  private static String filenameFlatten(String name) {
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
  
  private static String readUntilEndOfComment(Reader R)
      throws java.io.IOException {
    StringBuffer sb = new StringBuffer();
    
    int cha = R.read();
    while (cha != -1) {
      int chb = R.read();
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
        return ch >= 'a' && ch <= 'z' || ch >= 'A' && ch <= 'Z';
    }
    
  static Hashtable keywords = new Hashtable();
    
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
    static private Vector dat;

    static public void Sort(Vector data) {
        dat = data;
        sort(0, dat.size() - 1);
    }

    static private final long compare(Object a, Object b) {
        if (a instanceof Package)
            return ((Package)a).name.compareTo(((Package)b).name);
        else
            return ((JFile)a).name.compareTo(((JFile)b).name);
    }

    static private void sort(int p, int r) {
        if (p < r) {
            int q = partition(p,r);
            if (q == r) 
                q--;
            sort(p,q);
            sort(q+1,r);
        }
    }

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
  private boolean inCodeSegment = false;
  private PrintWriter pw;
  private long lineNumber = 1;
  private boolean atBeginningOfLine = true;

  LineNumberPrintWriter(PrintWriter pw) {
    this.pw = pw;
  }

  public void enterCodeSegment() {
    inCodeSegment = true;
    atBeginningOfLine = true;
  }
  
  public void exitCodeSegment() {
    if (!atBeginningOfLine) {
      lineNumber++;
      atBeginningOfLine = true;
    }
    inCodeSegment = false;
  }

  public void print(String s) {
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
    pw.print(ch);
    if (inCodeSegment && ch == '\n') {
      atBeginningOfLine = true;
      lineNumber++;
    }
  }

  public void println(String s) {
    print(s);
    print('\n');
  }

  public void close() {
    pw.close();
  }
}
