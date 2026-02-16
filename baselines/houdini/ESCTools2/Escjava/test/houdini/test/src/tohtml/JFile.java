package tohtml;

import java.io.*;
import java.util.*;

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
