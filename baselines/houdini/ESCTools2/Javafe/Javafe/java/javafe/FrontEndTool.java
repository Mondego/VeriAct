/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe;

import javafe.ast.PrettyPrint;
import javafe.ast.StandardPrettyPrint;

import javafe.reader.StandardTypeReader;
import javafe.parser.PragmaParser;

import javafe.tc.OutsideEnv;
import javafe.tc.TypeCheck;

import javafe.util.*;

import java.util.ArrayList;

/**
 * <code>FrontEndTool</code> is an abstract class for tools that use
 * our Java front end.
 *
 * <p> It handles parsing the standard options for setting up the
 * front end and initializing the front end using those options.  At
 * the end of a run, it prints a count of how many cautions, warnings,
 * and errors occurred (cf. <code>ErrorSet</code>).  It also handles
 * catching <code>FatalError</code>s (see
 * <code>ErrorSet.fatal</code>).  The remaining processing, if any, is
 * front-end-tool specific. </p>
 */

public abstract class FrontEndTool extends Tool
{
  /***************************************************
   *                                                 *
   * Standard front-end setup:		             *
   *                                                 *
   **************************************************/

  /**
   * Setup: initialize the front end using the standard
   * front-end-tool option variables ({@link Options#userPath}, 
   * {@link Options#sysPath}).
   *
   * <p> This can be done only once.  The standard front-end-tool
   * option variables have no effect after this point.  May exit
   * with an error (via {@link ErrorSet#fatal(String)}). </p>
   *
   * <p> Ensures {@link OutsideEnv} has been properly initialized
   * (except if an error occurs). </p>
   *
   * <p> Also initializes {@link PrettyPrint#inst} and {@link
   * TypeCheck#inst} to their default front end values. </p>
   */
  protected String compositeSourcePath;
  protected String compositeClassPath;

  public void setupPaths() {
    String classPath = getOptions().userPath;
    if (classPath == null)
      // The behavior of this code differs between 1.1 and 1.2:
      classPath = javafe.filespace.ClassPath.current();

    String sourcePath = getOptions().userSourcePath;

    String sys = getOptions().sysPath;
    if (sys == null) {
      // This works only on Sun implementations of Java...
      sys = System.getProperty("sun.boot.class.path", null);
      //System.out.println("SYS-SUN " + sys);
    }
    Info.out("sysPath: " + sys);
    if (sys == null) {
      sys =
        System.getProperty("java.home")
        + java.io.File.separator
        + "lib"
        + java.io.File.separator
        + "rt.jar";
      //System.out.println("SYS-JH " + sys);
    }

    // TODO: does this work if classPath is empty
    Info.out("sysPath: " + sys);
    if (sys != null && !sys.equals("")) {
      if (!classPath.equals("")) {
        classPath += java.io.File.pathSeparator;
      }
      classPath += sys;
    }
    compositeSourcePath = sourcePath;
    compositeClassPath = classPath;
  }

  public void setup() {
    // javafe.util.Info.on = options.v;
    setupPaths();
    Info.out("[Full classpath is " + compositeClassPath + "]");
    Info.out("[Full sourcepath is " + compositeSourcePath + "]");

    // It is ok if sourcePath is null; it then shares a database
    // of the contents of the directory path with classpath,
    // rather than creating a separate, identical database.
    OutsideEnv.init(
                    makeStandardTypeReader(
                                           compositeClassPath,
                                           compositeSourcePath,
                                           makePragmaParser()));

    PrettyPrint.inst = makePrettyPrint();
    TypeCheck.inst = makeTypeCheck();
  }

  /**
   * Called to clear any static initializations, so that the parser
   * can be called multiple times within one process.  Called as
   * part of construction of a new Main.
   */
  public void clear(boolean complete) {
    ErrorSet.clear();
    // FIXME LocationManagerCorrelatedReader.clear();
    OutsideEnv.clear();
  }

  /**
   * Called to obtain the {@link StandardTypeReader} to be used for
   * locating and reading in types.
   */
  public /*@non_null*/StandardTypeReader makeStandardTypeReader(
                                                   String path,
                                                   String sourcePath,
                                                   PragmaParser P) {
    return StandardTypeReader.make(path, sourcePath, P);
  }

  /**
   * Called to obtain the pragma parser to be used for parsing input
   * files.  If <code>null</code> is returned, then no pragma
   * parsing is done.  (By default, returns <code>null</code>).
   */
  public /*@nullable*/PragmaParser makePragmaParser() {
    return null;
  }

  /**
   * Called to create a new {@link Options} object.
   */
  public /*@non_null*/Options makeOptions() {
    return new Options();
  }

  /** Processes the options into the current Options instance as
   *  contained in the options field.
   * @param args The command-line arguments to process
   * @throws UsageError if the sequence of command-line arguments 
   * 			  is invalid
   */
  //+@ requires options != null;
  public void processOptions(/*@non_null*/String[/*#@non_null*/] args) throws UsageError {
    getOptions().processOptions(args);
  }

  /**
   * Called to obtain the pretty printer to set {@link
   * PrettyPrint#inst} to.  May not return <code>null</code>.  By
   * default, returns {@link javafe.ast.StandardPrettyPrint}.
   */
  public /*@non_null*/PrettyPrint makePrettyPrint() {
    return new StandardPrettyPrint();
  }

  /**
   * Called to obtain an instance of the {@link javafe.tc.TypeCheck}
   * class (or a subclass thereof) to be used for typechecking. May
   * not return <code>null</code>.  By default, returns {@link
   * javafe.tc.TypeCheck}.
   */
  public /*@non_null*/TypeCheck makeTypeCheck() {
    return new TypeCheck();
  }

  /***************************************************
   *                                                 *
   * Main processing code:			       *
   *                                                 *
   **************************************************/

  /**
   * Start up an instance of this tool using command-line arguments
   * <code>args</code>.
   *
   * <p> <strong>Note</strong>: this code needs to be copied
   * verbatim to each subclass of {@link Tool} except with the name
   * of the actual subclass inserted after the new operator and the
   * comment characters (//) removed. </p>
   *
   * <p> (This needs to be done because static methods cannot be
   * inherited.) </p>
   */
  //@ requires \nonnullelements(args);
  public static void main(/*@non_null*/String[/*#@non_null*/] args) {
    // Tool t = new FrontEndTool();
    // int result = t.run(args);
    // if (result != 0) System.exit(result);
  }

  /**
   * Parses the options into a new instance of an {@link Options}
   * subclass.  The new instance is assigned to the options field.
   * If the argument is null, the tool is initialized with the
   * existing options (the options field is unchanged).  The tool
   * is then initialized (by calling setup) with the designated options.
   * 
   * @param args Either null or an array of arguments used to 
   *             initialize the Options structure
   * @return     Returns -1 if arguments were parsed satisfactorily,
   * 		   otherwise returns the exit code with which to 
   * 		   terminate the program
   */

  //@ ensures args == null ==> \not_modified(options);
  // FIXME //@ ensures args == null ==> \not_modified(options.*);
  public int handleOptions(/*@nullable*/String[/*#@non_null*/] args) {
    if (args != null) {
      try {
        // Handle all tool options:
        options = makeOptions();
        processOptions(args);
        if (getOptions().issueUsage) {
          usage();
          return okExitCode;
        }
      } catch (UsageError e) {
        badOptionUsage(e);
        ErrorSet.errors++; // Just so that the JUnit tests detect that
        // an error message was issued
        return badUsageExitCode;
      } catch (FatalError e) {
        Info.out("[" + name() + " exiting due to a fatal error]");
      }
    }
    // Setup the front end using the options:
    setup();
    return -1;
  }

  /**
   * A tool's main entry point, which should be overridden in derived classes
   * to do the work of the tool.
   * 
   * @param args The command-line arguments the program was invoked with
   * @return The exit code for the program, with 0 indicating success
   * @see javafe.Tool#run(java.lang.String[])
   */
  /*@ also public normal_behavior
    @  modifies \everything;
    @*/
  public final int run(/*@non_null*/String[/*#@non_null*/] args) {
    int r = handleOptions(args);
    virtualMachineVersionCheck();
    if (r != -1)
      return r;

    if (ErrorSet.errors == 0)
      try {
        // Do our front-end-tool-specific processing:
        frontEndToolProcessing(getOptions().inputEntries);
      } catch (FatalError e) {
        Info.out("[" + name() + " exiting due to a fatal error]");
      }

    if (ErrorSet.cautions != 0)
      System.out.println(
                         ErrorSet.cautions
                         + " caution"
                         + (ErrorSet.cautions > 1 ? "s" : ""));
    if (ErrorSet.warnings != 0)
      System.out.println(
                         ErrorSet.warnings
                         + " warning"
                         + (ErrorSet.warnings > 1 ? "s" : ""));
    if (ErrorSet.errors != 0)
      System.out.println(
                         ErrorSet.errors + " error" + (ErrorSet.errors > 1 ? "s" : ""));

    // If we call exit here, we will break GUI-based clients.
    // Return error status to caller:
    if (ErrorSet.errors > 0)
      return errorExitCode;
    else {
      return okExitCode;
    }
  }

  /** Cache the fact that we already checked and warned the user. */
  private static boolean checkedAboutVM = false;
  /**
   * Check that we are running in the proper version of the Java VM.
   * The Java front-end works in all VM versions, thus the method body
   * here at this time is just a skip.
   */
  protected void virtualMachineVersionCheck() {
    if (checkedAboutVM)
      return;
    checkedAboutVM = true;
  }

  /**
   * Perform any front-end-tool-specific processing.
   *
   * <p> The remaining arguments are <code>args[offset]</code>,
   * <code>args[offset+1]</code>, ...</p>
   */
  // requires \nonnullelements(args);
  // requires 0 <= offset && offset <= args.length;
  public abstract void frontEndToolProcessing(/*@non_null*/ArrayList args);
}
