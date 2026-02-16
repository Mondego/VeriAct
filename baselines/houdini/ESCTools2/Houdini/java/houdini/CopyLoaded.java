/* Copyright 2000, 2001, Compaq Computer Corporation */

/** CopyLoaded loads the files specified by the '-f' parameter
  and copies it and its dependencies.

  By setting OutsideEnv.setListener(this), notify() is called
  every time a new CompilationUnit is loaded. The file 
  corresponding to the CompilationUnit is copied or parsed and
  prettyprinted as a spec file, and it is added to the 'loaded'
  stack.

  All CompilationUnits are popped from the loaded stack and passed
  to handleCU (which might cause more CompilationUnits to be loaded
  and thus be pushed onto loaded). Non-specOnly TypeDecl elements of 
  the CU will be handled further by handleTD.

  The popped CU's are added to the "handled" HashSet to prevent 
  loops - though it is not clear if that can ever happen in
  practice.

  This process ends when the loaded stack is empty after handling
  its last CU.
  
  It's not entirely clear to me yet, but it looks like classes which
  are merely used as parameters for methods for which only specs
  are available are not loaded, which leads to errors when trying
  to parse the copied code. 
*/

package houdini;


import java.util.Vector;
import java.util.HashSet;
import java.util.Stack;

import javafe.ast.*;
import javafe.tc.*;
import javafe.util.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.*;

import java.util.StringTokenizer;
import java.util.ArrayList;
import java.util.Iterator;

public class CopyLoaded extends javafe.FrontEndTool implements Listener {

    public String name() { return "CopyLoaded"; }

    private PrintWriter libIndirectWriter;
    private PrintWriter progIndirectWriter;

    /* 
     * This maps files from original progIndirect file names to their names 
     * relative to the outDir
     */
    public final Vector progIndirectFiles = new Vector();
    //+@ invariant progIndirectFiles.elementType == \type(String);
    // @  invariant progIndirectFiles.owner == this;
    //+@ invariant !progIndirectFiles.containsNull;

    public void setupPaths() {
	super.setupPaths();
	if (options().specspath == null) return;
	if (compositeSourcePath == null) {
	    compositeClassPath = 
		options().specspath
		+	java.io.File.pathSeparator
		+	compositeClassPath;
	} else {
	    compositeSourcePath = 
		options().specspath
		+	java.io.File.pathSeparator
		+	compositeSourcePath;
	}
    }

    /***************************************************
     *                                                 *
     * Keeping track of loaded CompilationUnits:       *
     *                                                 *
     **************************************************/


    // CU's that have been loaded, but not yet handled
    //@  invariant loaded != null;
    //+@ invariant loaded.elementType == \type(CompilationUnit);
    //+@ invariant !loaded.containsNull;
    // @  invariant loaded.owner == this;
    public Stack loaded = new Stack();

    // CU's that have already been handled
    public HashSet handled = new HashSet();

    // @ invariant loaded != argumentFileNames;
 
    public CopyLoaded() {
        //+@ set argumentFileNames.elementType = \type(String);
        //+@ set argumentFileNames.containsNull = false;
        // @  set argumentFileNames.owner = this;
    
        //+@ set loaded.elementType = \type(CompilationUnit);
        //+@ set loaded.containsNull = false;
        // @  set loaded.owner = this;
    
        //+@ set progIndirectFiles.containsNull = false;
        //+@ set progIndirectFiles.elementType = \type(String);
        // @  set progIndirectFiles.owner = this;
    }


    public void setup() { 
        super.setup();
    }


    private String packageDirForFile(/*@ non_null */ CompilationUnit cu) {
        Name pkg = cu.pkgName;
        String s = (pkg == null) ? "" : (pkg.printName() + ".");
        s = s.replace('.', '/'); 
        return s;
    }

    public void notify(CompilationUnit justLoaded) {
        // REVIEW: can this ever occur? if not, the 'handled' set
        // can be entirely removed.
        if (! handled.contains(justLoaded) )
          loaded.push(justLoaded);
     
        String fileName = Location.toFileName(justLoaded.loc);
        /* if a Java file, then copy the file over into outDir */
        if (fileName.endsWith(".java") || fileName.endsWith(".spec") || fileName.endsWith(".jml")) {
            copySourceFile(fileName, 
                   packageDirForFile(justLoaded) + fileNameName(fileName));
        } else {
            TypeDeclVec elems = justLoaded.elems;
            /* generate a spec file for each type decl in compilation unit */
            for (int i=0; i<elems.size(); i++) {
                TypeDecl d = elems.elementAt(i);
                TypeSig sig = TypeCheck.inst.getSig(d);
                if (d.specOnly || d.isBinary()) {
                    printSpec(sig.toString());
                }
            }
        }
    }

    /***************************************************
     *                                                 *
     * Main processing code:                   *
     *                                                 *
     **************************************************/

    //@ requires \nonnullelements(args);
    public static void main(String[] args) {
        javafe.Tool t = new CopyLoaded();
        int result = t.run(args);
        if (result != 0) System.exit(result);
    }
    
    public javafe.Options makeOptions() {
        return new CopyLoadedOptions();
    }
    
    public CopyLoadedOptions options() {
        return (CopyLoadedOptions)options;
    }



    //@ ensures \nonnullelements(\result);
    //@ ensures \result != null;
    public String[] FQNpackage(/*@ non_null */ String s) {
        StringTokenizer st = new StringTokenizer(s, ".", false);
        int len = st.countTokens();
        return fillArray(st, len-1);
    } 

    //@ ensures \result != null;
    public String FQNname(/*@ non_null */ String s) {
         return s.substring(s.lastIndexOf(".") + 1);
    } 
 
    //@ ensures \nonnullelements(\result);
    //@ ensures \result != null;
    public String[] fileNamePackage(/*@ non_null */ String file) {
        StringTokenizer st = new StringTokenizer(file, "/", false);
        int len = st.countTokens();
        return fillArray(st, len-1);
    } 


    //@ ensures \result != null;
    public String fileNameName(/*@ non_null */ String s) {
         return s.substring(s.lastIndexOf("/") + 1);
    }    

    //@ ensures \nonnullelements(\result);
    //@ ensures \result != null;
    public String[] directoryPackage(/*@ non_null */ String dir) {
        StringTokenizer st = new StringTokenizer(dir, "/", false);
        int len = st.countTokens();
        return fillArray(st, len);
    } 

    //@ ensures \nonnullelements(\result);
    //@ ensures \result != null;
    private String[] fillArray(/*@ non_null */ StringTokenizer st, int len) {
        if (len < 0) {
            return new String[0];
        }
        String array[] = new String[len];
        for (int i = 0; i < len; i++) {
            array[i] = st.nextToken();
        }
        return array;
    }    

    //@ requires \nonnullelements(P);
    private String makeDirTree(/*@ non_null */ String root, 
                               /*@ non_null */ String P[]) {
        String s = root;
        for (int i = 0; i < P.length; i++) {
            s = s + "/" + P[i];
            File f = new File(s);
            if (!f.exists()) {
                System.out.println("[making " + s + "]");
                f.mkdir();            
            } else {
            }
        }
        return s;
    }

    //@ requires \nonnullelements(P);
    //@ ensures \result != null;
    private String makeDirPath(/*@ non_null */ String P[]) {
        String s = "";
        for (int i = 0; i < P.length; i++) {
            s = s + P[i] +"/";
        }
        return s;
    }

    /**
     * Prints the spec file for the FQN s.  The file is written
     * relative to the outDir.
     */  
    public void printSpec(/*@ non_null */ String s) {
        String P[] = FQNpackage(s);
        String T = FQNname(s);
        TypeSig sig = OutsideEnv.lookup(P, T);
        Assert.notFalse(sig != null); //@ nowarn Pre;
    
        String path = makeDirTree(options().outDir, P);
        String outFile = T + ".spec";
        String filename = path + "/" +  outFile;
    
        System.out.println("[generating spec file for " + s + 
                           " as " + filename + "]");
    
        //@ assume libIndirectWriter != null;
        libIndirectWriter.println("./" + makeDirPath(P) + outFile);
    
        try {
            FileOutputStream fos = null;
            try {
                fos = new FileOutputStream(filename);
		CompilationUnit c = sig.getCompilationUnit();
                PrettyPrint.inst.print(fos, c);
            } finally {
                if (fos != null) fos.close();
            }
        } catch (IOException e) {
	    //System.out.println (e.toString());
            ErrorSet.fatal(e.getMessage());
        }
/*
        } catch (Exception e) {
	    System.out.println (e.toString());
            ErrorSet.fatal(e.getMessage());
        }
*/
    }

    public final void frontEndToolProcessing(ArrayList args) {
        /*
         * At this point, all options have already been processed and
         * the front end has been initialized.
         */
    
        String outDir = options().outDir;
        String outProgIndirect = options().outProgIndirect;
        String outLibIndirect = options().outLibIndirect;
        
        System.out.println("[outdir is " + outDir + "]");
    
        // Set up to receive CompilationUnit-loading notification events:
        OutsideEnv.setListener(this);
    
        // create the outDir
        if (outDir.startsWith(".")) {
            makeDirTree(".", directoryPackage(outDir));
        } else {
            makeDirTree("", directoryPackage(outDir));
        } 
        
        // set up the indirection files.
        try {
            progIndirectWriter =
              new PrintWriter(new FileWriter(new File(outProgIndirect)));
            libIndirectWriter = 
              new PrintWriter(new FileWriter(new File(outLibIndirect)));
        } catch (IOException e) {
            ErrorSet.fatal(e.getMessage());
        }
    
        /*
         * Load in each source file:
         */
	Iterator i = args.iterator();
	while (i.hasNext()) {
	    String next = ((javafe.InputEntry)i.next()).toString();
	    OutsideEnv.addSource(next);
	}
    
        /* load in source files from supplied file name */
	i = options().argumentFileNames.iterator();
	while (i.hasNext()) {
	    String argumentFileName = (String)i.next();
	    try {
		    BufferedReader in = null;
	        try {
	            in = new BufferedReader(
	                    new FileReader(argumentFileName));
	            String s;
	            while ((s = in.readLine()) != null) {
	                // allow blank lines in files list
	                if (!s.equals("")) {
	                    progIndirectFiles.addElement(s);
	                    OutsideEnv.addSource(s); 
	                }
	            }
	        } finally {
	            if (in != null) in.close();
	        }
	    } catch (IOException e) {
	        ErrorSet.fatal(e.getMessage());
	    }
	}
    
        while (!loaded.empty()) {
            CompilationUnit nextcu = (CompilationUnit)loaded.pop();
            handled.add(nextcu);
            handleCU(nextcu);
        }
    
        progIndirectWriter.close();
        libIndirectWriter.close();
    }


    /**
     * Copy the source file original into the file newName.  newName
     * is appended to the outDir to construct the full file location
     * of the new file.  This method also puts the newName into the
     * correct indirection file.
     */
    private void copySourceFile(/*@ non_null */ String original, 
                /*@ non_null */ String newName) {
        try {
            String path = makeDirTree(options().outDir, fileNamePackage(newName));
            String newFileName = path + "/" + fileNameName(newName);
    
            System.out.println("[copying source file " + original + 
                       " to " + newFileName + "]");
    
            //@ assume libIndirectWriter != null;
            //@ assume progIndirectWriter != null;
            if (progIndirectFiles.contains(original)) {
                progIndirectWriter.println("./" + newName);
            } else {
                libIndirectWriter.println("./" + newName);
            }
            
            File f = new File(newFileName);
            BufferedReader reader = null;
            PrintWriter writer = null;
            try {
                reader = new BufferedReader(new FileReader(original));
                try {
                    writer = new PrintWriter(new FileWriter(f));
                    String s;
                    while ((s = reader.readLine()) != null) {
                        writer.println(s);
                    }
                } finally {
                    if (writer != null) writer.close();
                }
            } finally {
                if (reader != null) reader.close();
            }
        } catch (IOException e) {
            ErrorSet.fatal(e.getMessage());
        }
    }


    /**
     * Process each CU's type decls.
     */
    public void handleCU(/*@ non_null */ CompilationUnit cu) {
        // Iterate over all the TypeDecls representing outside types in cu:
        TypeDeclVec elems = cu.elems;
        for (int i=0; i<elems.size(); i++) {
            TypeDecl d = elems.elementAt(i);
            handleTD(d);
        }
    }

    /**
     * Called from handleCU on each TypeDecl from the CU's loaded from the
     * program files.  In addition, it calls itself recursively to handle types
     * nested within outside types.<p>
     */
    public void handleTD(/*@ non_null */ TypeDecl td) {
        TypeSig sig = TypeCheck.inst.getSig(td);
        if (sig.getTypeDecl().specOnly)    // do not process specs
            return;
        
        System.out.println("\n" + sig.toString() + " ...");
        
                    // Do actual work:
        boolean aborted = processTD(td);
        
        TypeDecl decl = sig.getTypeDecl();
        for (int i=0; i<decl.elems.size(); i++) {
            if (decl.elems.elementAt(i) instanceof TypeDecl)
            handleTD((TypeDecl)decl.elems.elementAt(i)); //@ nowarn Cast;
        }
    }
    
    
    /**
     * Typecheck a TypeDecl;
     * return true if we had to abort. <p>
     *
     * Precondition: td is not from a binary file.<p>
     */
    private boolean processTD(/*@ non_null */ TypeDecl td) {
        int errorCount = ErrorSet.errors;
        TypeSig sig = TypeCheck.inst.getSig(td);
        sig.typecheck();
    
        return false;
    }
    

    /**
     ** Returns the Esc StandardTypeReader, EscTypeReader.
     **/
    public javafe.reader.StandardTypeReader makeStandardTypeReader(String classpath,
						     String sourcepath,
						     javafe.parser.PragmaParser P) {
        // parallel to code in Escjava/java/escjava/Main.java
        return escjava.reader.EscTypeReader.make(classpath, sourcepath, P, new escjava.AnnotationHandler ());
    }

    /**
     ** Returns the EscPragmaParser.
     **/
    public javafe.parser.PragmaParser makePragmaParser() {
      return new escjava.parser.EscPragmaParser();
    }

    /**
     ** Returns the pretty printer to set
     ** <code>PrettyPrint.inst</code> to.
     **/
    public PrettyPrint makePrettyPrint() {
      DelegatingPrettyPrint p = new escjava.ast.EscPrettyPrint();
      p.setDel(new StandardPrettyPrint(p));
      return p;
    }

    /**
     ** Called to obtain an instance of the javafe.tc.TypeCheck class
     ** (or a subclass thereof). May not return <code>null</code>.  By
     ** default, returns <code>javafe.tc.TypeCheck</code>.
     **/
    public javafe.tc.TypeCheck makeTypeCheck() {
	return new escjava.tc.TypeCheck();
    }
 
}

class CopyLoadedOptions extends escjava.Options {

    /*@ non_null */ public String outDir = "./outdir/src-annotated";
    /*@ non_null */ public String outProgIndirect = "./outProgIndirect";
    /*@ non_null */ public String outLibIndirect = "./outLibIndirect";

    public final Vector argumentFileNames = new Vector();
    //+@ invariant argumentFileNames.elementType == \type(String);
    //+@ invariant !argumentFileNames.containsNull;
    // @  invariant argumentFileNames.owner == this;
   
    public String showNonOptions() {
        return ("<program indirection file>");
    }

    public String showOptions(boolean all) {
        StringBuffer sb = new StringBuffer(super.showOptions(all));
        sb.append("  -outdir <root of output files>"); sb.append(eol);
        sb.append("  -outProgIndirect <new prog Indirect file>");sb.append(eol);
        sb.append("  -outLibIndirect <new lib Indirect file>"); sb.append(eol);
        sb.append("  -f <file containing source file names>"); sb.append(eol);
        return sb.toString();
    }


    /***************************************************
     *                                                 *
     * Option processing:                   *
     *                                                 *
     **************************************************/

    //private final String name = "CopyLoaded";
    
    public int processOption(String option, String[] args, int offset) 
                                        throws UsageError {
        if (option.equals("-f")) { 
            if (offset>=args.length) {
                throw new UsageError("Option " + option + 
                                         " requires one argument");
            }
            argumentFileNames.addElement(args[offset]);
            return offset + 1;
        } else if (option.equals("-outProgIndirect")) {
            if (offset>=args.length) {
                throw new UsageError("Option " + option + 
                                         " requires one argument");
            }
            outProgIndirect = args[offset];
            return offset + 1;
        } else if (option.equals("-outLibIndirect")) {
            if (offset>=args.length) {
                throw new UsageError("Option " + option + 
                                         " requires one argument");
            }
            outLibIndirect = args[offset];
            return offset + 1;
        } else if (option.equals("-outdir")) {
            if (offset>=args.length) {
                throw new UsageError("Option " + option + 
                                         " requires one argument");
            }
            outDir = args[offset];
            return offset + 1;
        } else {
            // Pass on unrecognized options:
            return super.processOption(option, args, offset);
        }
    }
}
