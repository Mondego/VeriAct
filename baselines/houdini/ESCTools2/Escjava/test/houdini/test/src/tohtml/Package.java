package tohtml;

import java.io.*;
import java.util.*;

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
