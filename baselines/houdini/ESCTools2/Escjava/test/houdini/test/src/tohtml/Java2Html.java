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
