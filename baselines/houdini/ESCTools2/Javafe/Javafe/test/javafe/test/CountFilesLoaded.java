/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.test;


import javafe.Tool;
import javafe.SrcTool;
import javafe.ast.TypeDecl;
import javafe.tc.TypeCheck;
import javafe.tc.OutsideEnv;
import javafe.util.UsageError;


/** This class reports the number of files loaded after all parsing is completed.
 **/

public class CountFilesLoaded extends SrcTool {

    /***************************************************
     *                                                 *
     * Generating a usage message:		       *
     *                                                 *
     ***************************************************/

    /**
     ** Return the name of this tool.  E.g., "ls" or "cp".<p>
     **
     ** Used in usage and error messages.<p>
     **/
    public String name() { return "CountFilesLoaded"; }

    public javafe.Options makeOptions() { return new Options(); }
    
    public class Options extends javafe.SrcToolOptions {
    
        public Options() {
            OutsideEnv.eagerRead = false;
        }
    
	public String showOptions(boolean all) {
	    return super.showOptions(all) +("  -eager")+eol;
	}


	/***************************************************
	 *                                                 *
	 * Generic option processing:		       *
	 *                                                 *
	 ***************************************************/
    
	/**
	 ** Process next tool option. <p>
	 **
	 ** See <code>Tool.processOption</code> for the complete
	 ** specification of this routine.<p>
	 **/
	public int processOption(String option, String[] args, int offset) 
                                 throws UsageError {
	    if (option.equals("-eager")) {
		OutsideEnv.eagerRead = true;
		return offset;
	    }
	
	    // Pass on unrecognized options:
	    return super.processOption(option, args, offset);
	}
    }

    /***************************************************
     *                                                 *
     * Main processing code:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Start up an instance of this tool using command-line arguments
     ** <code>args</code>. <p> 
     **/
    //@ requires \nonnullelements(args);
    public static void main(String[] args) {
	Tool t = new CountFilesLoaded();
	int result = t.run(args);
	if (result != 0) System.exit(result);
    }


    /***************************************************
     *                                                 *
     * SrcTool-instance specific processing:	       *
     *                                                 *
     ***************************************************/

    /**
     ** Hook for any work needed after <code>handleCU</code> has been called
     ** on each <code>CompilationUnit</code> to process them.
     **/
    public void postprocess() {
	System.out.println("Files parsed/loaded = " + OutsideEnv.filesRead());
    }


    /**
     ** This method is called on the TypeDecl of each
     ** outside type that SrcTool is to process. <p>
     **/
    public void handleTD(TypeDecl td) {
	if (!td.specOnly)
		TypeCheck.inst.checkTypeDecl(td);
    }
}
