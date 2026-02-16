/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.test;

import java.util.Enumeration;
import java.util.Vector;

import javafe.*;
import javafe.ast.*;
import javafe.tc.*;
import javafe.parser.*;
import javafe.util.*;


/**
 ** <code>Print</code> is a <code>SrcTool</code> that parses input
 ** file(s) and pretty prints them, optionally doing typechecking on the
 ** type declarations inside of them.
 **/

public class Print extends SrcTool {
  
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
  public String name() { return "Print"; }
  
  public javafe.Options makeOptions() { return new Options(); }
  
  public final Options options() { return (Options)options; }
  
  public class Options extends SrcToolOptions {
    
    public Options() {
      PrettyPrint.displayInferred = false;
    }
    
    /**
     ** Print option information to
     ** <code>System.err</code>. <p>
     **/
    public String showOptions(boolean all) {
      return super.showOptions(all) +
      "  -typecheck" + eol +
      "  -noprint" + eol +
      "  -printtype" + eol +
      "  -showinferred" + eol;
    }
    
    
    /***************************************************
     *                                                 *
     * Option processing:			       *
     *                                                 *
     ***************************************************/
    
    private boolean typecheck = false;
    private boolean print = true;
    private boolean printType = false;
    // -showinferred controls PrettyPrint.displayInferred
    
    
    /**
     ** Process next tool option. <p>
     **
     ** See <code>Tool.processOption</code> for the complete
     ** specification of this routine.<p>
     **/
    public int processOption(String option, String[] args, int offset) 
    throws UsageError {
      if (option.equals("-typecheck")) {
        typecheck = true;
        return offset;
      } else if (option.equals("-noprint")) {
        print = false;
        return offset;
      } else if (option.equals("-printtype")) {
        printType = true;
        return offset;
      } else if (option.equals("-showinferred")) {
        PrettyPrint.displayInferred = true;
        return offset;
      }
      
      // Pass on unrecognized options:
      return super.processOption(option, args, offset);
    }
  }
  
  
  /***************************************************
   *                                                 *
   * Front-end setup:				                   *
   *                                                 *
   ***************************************************/
  
  /**
   ** Setup: see FrontEnd.setup for original spec. <p>
   **
   ** Print overrides this method to handle printtype.
   **/
  public void setup() {
    super.setup();
    
    if (options().printType) {
      TypePrint T = new TypePrint();
      PrettyPrint.inst = new StandardPrettyPrint(T);
      T.setDel(PrettyPrint.inst);
      // as required by TypePrint()
    }
  }
  
  
  /***************************************************
   *                                                 *
   * Main processing code:			               *
   *                                                 *
   ***************************************************/
  
  /**
   ** Start up an instance of this tool using command-line arguments
   ** <code>args</code>. <p> 
   **
   ** <b>Note</b>: this code needs to be copied verbatim to each
   ** subclass of <code>Tool</code> except with the name of the actual
   ** subclass inserted after the new operator and the comment
   ** characters (//) removed.<p>
   **
   ** (This needs to be done because static methods cannot be
   ** inherited.)<p>
   **/
  //@ requires \nonnullelements(args);
  public static void main(String[] args) {
    Tool t = new Print();
    int result = t.run(args);
    if (result != 0) System.exit(result);
  }
  
  
  /***************************************************
   *                                                 *
   * SrcTool-instance specific processing:	           *
   *                                                 *
   ***************************************************/
  
  /**
   ** This method is called on each <code>CompilationUnit</code>
   ** that this tool processes. <p>
   **/
  public void handleCU(CompilationUnit cu) {
    if (!options().typecheck)
      System.out.println("=== File: " + Location.toFileName(cu.loc)
                         +" ===");
    super.handleCU(cu);
    if (options().print) PrettyPrint.inst.print(System.out, cu);
  }
  
  
  /**
   ** This method is called on the TypeDecl of each
   ** outside type that SrcTool is to process. <p>
   **/
  public void handleTD(TypeDecl td) {
    td.check();
    if (options().typecheck && !td.specOnly) {
      TypeCheck.inst.checkTypeDecl(td);
      TypeSig sig = TypeSig.getSig(td);
      Assert.notFalse(sig.state == TypeSig.CHECKED);  //@ nowarn Pre;
      if (ErrorSet.errors == 0) {
        sig.deepCheck();
        td.check();
      }
    }
  }
}
