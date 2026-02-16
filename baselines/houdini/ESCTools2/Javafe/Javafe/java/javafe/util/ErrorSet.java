/* Copyright 2000, 2001, Compaq Computer Corporation */
// Modifications Copyright 2004, David Cok.

package javafe.util;

import java.io.*;
import javafe.FrontEndTool;

/**
 * The <code>ErrorSet</code> class is responsible for displaying
 * cautions, warnings, ordinary errors, and fatal errors to the user.
 * It maintains counts of how many cautions, warnings, and errors have
 * been reported so far.
 *
 * <p> Currently, reporting is done via printing all messages to
 * <code>System.out</code>.  Messages are accompanied by an indication
 * of whether they are a caution, a warning, an ordinary error, or a
 * fatal error.
 *
 * <p> Messages are printed as soon as they are reported.  A future
 * version of <code>ErrorSet</code> may group together messages
 * concerning the same file to be printed in location sorted order
 * once the file has been processed fully.  (This will require adding
 * some kind of <code>end_of_file(F)</code> call.)
 *
 * <p> Rough rules for determining what class to report something as:
 * <dl>
 * <dt> Error: <dd> something that is definitely wrong and which
 *	prevents processing everything the tool was requested to work
 *	on.
 *
 * <dt> Fatal Error: <dd> similar except that it prevents any further
 *      processing at all.  (Tools may continue after ordinary errors
 *      by aborting processing of some, but not all, parts of the
 *      input program.)
 *
 * <dt> Caution: <dd>
 *      (1) something that is technically illegal according to the
 *	language spec(s), but the tool accepts anyways for
 *	compatibility with other tools.  (It must not force
 *      aborting.) <br>
 *      (2) something that is not technically illegal, but is likely
 *	to be misleading. (The user is encouraged to adjust the
 *	environment or the code to remove the caution.)
 *
 * <dt> Warning: <dd>something that the Tool believes, but is not
 *	sure, is a serious problem - used for static checking reports.
 * </dl>
 *
 * @see Location
 * @see FatalError
 */

public class ErrorSet
{
  // The options field must be initialized before any calls here.
  //@ invariant FrontEndTool.options != null;

  // Prevent javadoc from displaying a public constructor
  private ErrorSet() {};


  /***************************************************
   *                                                 *
   * Class Variables:				       *
   *                                                 *
   **************************************************/

  /**
   * The number of cautions reported so far.
   */
  public static int cautions = 0;


  /**
   * The number of warnings reported so far.
   */
  public static int warnings = 0;


  /**
   * The number of errors reported so far, including fatal errors.
   */
  public static int errors = 0;

  /**
   * The number of fatal errors so far (some may have been caught and handled)
   */
  public static int fatals = 0;

  /**
   * If <code>gag</code> is true, then no output is produced by
   * <code>ErrorSet</code> methods (useful for test harnesses).
   *
   * Defaults to false.<p>
   */
  public static boolean gag = false;

  /** Resets all error and warning counts. */
  public static void clear() {
    fatals = 0;
    errors = 0;
    warnings = 0;
    cautions = 0;
    gag = false;
  }

  private static int savedErrorsWarnings;
  private static int savedCautions;

  public static void mark() {
    savedErrorsWarnings = errors + warnings;
    savedCautions = cautions;
  }

  public static boolean errorsSinceMark() {
    return errors + warnings > savedErrorsWarnings;
  }

  public static boolean cautionsSinceMark() {
    return cautions > savedCautions;
  }

  /***************************************************
   *                                                 *
   * Reporting entry points:			       *
   *                                                 *
   **************************************************/

  /**
   * Report a caution. <p>
   * 
   * The message will be marked as a caution when it is displayed to
   * the user.  Increments <code>cautions</code> by one.<p>
   */
  //@ requires msg != null;
  //@ modifies cautions, System.out.output;
  //@ ensures gag ==> \not_modified(System.out.output);
  //@ ensures FrontEndTool.options.noCautions ==> \not_modified(System.out.output);
  public static void caution(String msg) {
    if (FrontEndTool.options.noCautions)
      return;
    cautions++;
    report(CAUTION, msg);
  }

  /**
   * Report a caution associated with a location. <p>
   * 
   * Precondition: <code>loc</code> must be a regular location or a
   * whole file location.<p>
   *
   * The message will be marked as a caution when it is displayed to
   * the user.  Increments <code>cautions</code> by one.<p>
   */
  //@ requires loc != Location.NULL;
  //@ requires msg != null;
  //@ modifies cautions, System.out.output;
  public static void caution(int loc, String msg) {
    if (FrontEndTool.options.noCautions) {
      return;
    }
    cautions++;
    report(loc, CAUTION, msg);
  }

  //@ requires loc != Location.NULL;
  //@ requires msg != null;
  //@ modifies cautions, System.out.output;
  public static void caution(int loc, String msg, int addLoc) {
    if (FrontEndTool.options.noCautions) {
      return;
    }
    cautions++;
    report(loc, CAUTION, msg);
    assocLoc(addLoc);
  }


  /**
   * Report a warning. <p>
   * 
   * The message will be marked as a warning when it is displayed to
   * the user.  Increments <code>warnings</code> by one.<p>
   */
  //@ requires msg != null;
  //@ modifies warnings, System.out.output;
  public static void warning(String msg) {
    warnings++;
    report(WARNING, msg);
  }

  /**
   * Report a warning associated with a location. <p>
   * 
   * Precondition: <code>loc</code> must be a regular location or a
   * whole file location.<p>
   *
   * The message will be marked as a warning when it is displayed to
   * the user.  Increments <code>warnings</code> by one.<p>
   */
  //@ requires loc != Location.NULL;
  //@ requires msg != null;
  //@ modifies warnings, System.out.output;
  public static void warning(int loc, String msg) {
    warnings++;
    report(loc, WARNING, msg);
  }


  /**
   * Report an ordinary error. <p>
   * 
   * The message will be marked as an error when it is displayed to
   * the user.  Increments <code>errors</code> by one.<p>
   */
  //@ requires msg != null;
  //@ modifies errors, System.out.output;
  public static void error(String msg) {
    errors++;
    report(ERROR, msg);
  }

  /**
   * Report an ordinary error associated with a location. <p>
   * 
   * Precondition: <code>loc</code> must be a regular location or a
   * whole file location.<p>
   *
   * The message will be marked as an error when it is displayed to
   * the user.  Increments <code>errors</code> by one. <p>
   */
  //@ requires loc != Location.NULL;
  //@ requires msg != null;
  //@ modifies errors, System.out.output;
  public static void error(int loc, String msg) {
    errors++;
    report(loc, ERROR, msg);
  }

  //@ requires loc != Location.NULL;
  //@ requires msg != null;
  //@ modifies errors, System.out.output;
  public static void error(int loc, String msg, int addLoc) {
    errors++;
    report(loc, ERROR, msg);
    assocLoc(addLoc);
  }

  //@ modifies System.out.output;
  public static void assocLoc(int loc) {
    if (loc != Location.NULL)
      reporter.reportAssociatedInfo(loc);
  }

  /**
   * Report a fatal error.  Warning: This method does not
   * return normally!<p>
   * 
   * The variable <code>errors</code> is incremented by one, the
   * error reported as a fatal error, and then an unchecked
   * <code>FatalError</code> exception is thrown.  The top level of a
   * <code>Tool</code> is responsible for catching the
   * <code>FatalError</code>, so that it can do whatever cleanup is
   * required before exiting.<p>
   */
  //@ requires msg != null;
  //@ modifies fatals, errors, System.out.output;
  //@ ensures false;
  //@ signals (Exception) fatals == \old(fatals)+1;
  //@ signals (Exception) errors == \old(errors)+1;
  //@ signals_only FatalError;
  public static void fatal(String msg) /*throws FatalError*/ {
    if (msg != null) {
      fatals++;
      errors++;
      report(FATALERROR, msg);
    }
    throw new FatalError(); 
  }                 //@ nowarn Exception;

  /**
   * Report a fatal error associated with a location.  Warning: This
   * method does not return normally!<p>
   * 
   * Precondition: <code>loc</code> must be a regular location or a
   * whole file location.<p>
   *
   * The variable <code>errors</code> is incremented by one, the
   * error reported as a fatal error, and then an unchecked
   * <code>FatalError</code> exception is thrown.  The top level of a
   * <code>Tool</code> is responsible for catching the
   * <code>FatalError</code>, so that it can do whatever cleanup is
   * required before exiting.<p>
   */
  //@ requires loc != Location.NULL;
  //@ requires msg != null;
  //@ modifies fatals, errors, System.out.output;
  //@ ensures false;
  //@ signals (Exception) fatals == \old(fatals)+1;
  //@ signals (Exception) errors == \old(errors)+1;
  //@ signals_only FatalError;
  public static void fatal(int loc, String msg) /*throws FatalError*/ {
    fatals++;
    errors++;
    report(loc, FATALERROR, msg);
    throw new FatalError(); 
  }				//@ nowarn Exception;

  /** Special call to report unimplemented features, so they can be caught
      and handled more easily.
  */
  // FIXME - do we need this?
  //@ requires msg != null;
  //@ requires loc != Location.NULL;
  //@ modifies System.out.output;
  //@ ensures false;
  //@ signals_only NotImplementedException;
  public static void notImplemented(boolean print, int loc, String msg) {
    if (print) report(loc, "Not implemented", msg);
    throw new NotImplementedException(msg); 
  } 				//@ nowarn Exception;

  /***************************************************
   *                                                 *
   * Common code for reporting:		       *
   *                                                 *
   **************************************************/

  // Constants for use as the type field of report:

  private static final /*@ non_null */ String CAUTION		= "Caution";
  private static final /*@ non_null */ String WARNING		= "Warning";
  private static final /*@ non_null */ String ERROR		= "Error";
  private static final /*@ non_null */ String FATALERROR	= "Fatal error";


  /**
   * Report general information.
   *
   * <p> Type contains a non-null String describing the type of the
   * information (usually one of the above constants).  The
   * information itself is contained in the non-null String
   * msg.
   *
   * <p> This function is not responsible for incrementing counts or
   * other ErrorSet functionality.
   */
  //@ requires msg != null && type != null;
  //@ requires javafe.Tool.options != null;
  //@ modifies System.out.output;
  //@ ensures gag ==> \not_modified(System.out.output);
  private static void report(/*@ non_null */ String type, /*@ non_null */ String msg) {
    if (!gag)
      report(type + ": " + msg);

    // Hack so we can see where error occurred, for debugging:
    if (javafe.Tool.options.showErrorLocation) dump(null); //@nowarn Modifies;
    // Ignore the modifications caused by the debugging line
	
  } //@ nowarn Post; // dump does not satisfy the postcondition

  /**
   *  Reports a general message, implemented here in order to
   *  have a single location through which error reporting happens.
   */
  //@ requires msg != null;
  //@ modifies System.out.output;
  public static void report(/*@ non_null @*/ String msg) {
    reporter.report(0,Location.NULL,-1,msg);
  }


  /**
   * Report information associated with a location. <p>
   *
   * Type contains a non-null String describing the type of the
   * information (usually one of the above constants).  The
   * information itself is contained in the non-null String
   * msg.<p>
   *
   * Precondition: <code>loc</code> must be a regular location or a
   * whole file location.<p>
   *
   * This function is not responsible for incrementing counts or
   * other ErrorSet functionality.<p>
   */
  //@ requires msg != null && type != null;
  //@ requires loc != Location.NULL;
  //@ modifies System.out.output;
  private static void report(int loc, String type, String msg) {
    if (gag)
      return;

    // Hack so we can see where error occurred, for debugging:
    if (javafe.Tool.options.showErrorLocation) dump(null); //@ nowarn Modifies;
    // Ignore the modifications caused by the debugging line

    if (loc==Location.NULL) {
      errors++;
      report("INTERNAL ERROR: ",
             "NULL location erroneously passed to ErrorSet;"
             + " Associated information is `" + type
             + ": " + msg + "'");
    }

    // Display the file human-readable name and line # if available:

    reporter.report(0,loc,-1,type + ": " + msg);

  }


  /***************************************************
   *                                                 *
   * Utility routines:			       *
   *                                                 *
   **************************************************/

  /**
   * Return a new InputStream for the file that loc refers to or null
   * if an I/O error occurs while attempting to open the stream. <p>
   *
   * Precondition: <code>loc</code> must be a regular location or a
   * whole file location.<p>
   *
   * No error is reported if an I/O error occurs.
   */
  //@ requires loc != Location.NULL;
  //---@ modifies \nothing;
  //@ ensures \result != null ==> \result.isOpen;
  //@ ensures \fresh(\result); // FIXME - not sure about this
  //@ signals_only \nothing;
  private static InputStream getFile(int loc) {
    try {
      return Location.toFile(loc).getInputStream();
    } catch (IOException e) {
      return null;
    }
  }


  /**
   * Return the line loc refers to or null if an I/O error occurs
   * while attempting to read the line in. <p>
   *
   * Precondition: <code>loc</code> is a regular location (e.g., not
   * a whole-file location).<p>
   *
   * No error is reported if an I/O error occurs.
   */
  //@ requires loc != Location.NULL;
  //@ modifies \nothing;
  //@ ensures \fresh(\result);
  //@ signals_only \nothing;
  private static String getLine(int loc) {
    InputStream i = getFile(loc);
    if (i==null)
      return null;

    // FIXME - why not use a buffered Reader here to read
    // characters?
    try {
      /*
       * skip to the start of the line in question:
       */
      long charsLeft = (Location.toOffset(loc)-1)
        - Location.toColumn(loc);
      // FIXME - wrong if offset is 0 - see if esc catches it
      while ((charsLeft -= i.skip(charsLeft))>0) { // skip all but 1 byte
        i.read();  // skip the last byte
        charsLeft--;
      }

      // FIXME - this seems awfully inefficient
      StringBuffer line = new StringBuffer(100);
      for (int c=i.read(); c != '\r' && c != '\n' && c!= -1/*EOF*/; c=i.read())
        line.append((char)c);

      i.close();
      return line.toString();
    } catch (IOException e) {
      return null;
    }
  }


  /** See documentation for two-argument version of <code>displayColumn</code>.
   * This version differs in that the default clip policy is applied.
   */
  //@ requires loc != Location.NULL;
  //@ requires !Location.isWholeFileLoc(loc);
  //@ modifies System.out.output;
  //@ ensures true;
  //@ signals_only \nothing;
  public static void displayColumn(int loc) {
    displayColumn(loc, null);
  }

  static private final int TABSTOP = 8;
  /**
   * Display (part of) the line that loc occurs on, then indicate via
   * a caret (^) which column loc points to. <p>
   *
   * Tabs are expanded before the line is displayed using 8-character
   * tab stops.  8 spaces is perhaps not what the user intended, but
   * there will not be any other lines portrayed against which it must
   * match, and the caret positioning will be consistent with the 8
   * character spacing; at worst, the line will be spread out more than
   * it would be with shorter tabs. <p>
   *
   * Precondition: <code>loc</code> is a regular location (e.g., not
   * a whole-file location).<p>
   *
   * If an I/O error does occur, then the user is informed of the
   * column number and that the line in question is not available; no
   * error is reported.
   *
   * By using a non-null <code>policy</code> argument, a caller can fine-
   * tune the policy used for introducing ellipses in the printed line.
   */
  //@ public normal_behavior
  //@ requires loc != Location.NULL;
  //@ requires !Location.isWholeFileLoc(loc);
  //@ modifies System.out.output;
  //@ ensures true;
  public static void displayColumn(int loc, ClipPolicy policy) {
    String line = getLine(loc);
    int col = Location.toColumn(loc);     // zero-based
    //@ assume line != null ==> (col >= 0 && col < line.length());  // FIXME
    if (line==null) {
      System.out.println("(line text not available; see column "
                         + col + ")");
      return;
    }


    /*
     * Expand tabs in line, keeping col in sync:
     */
    StringBuffer adjusted = new StringBuffer(100);
    int x = 0;		// tab-adjusted location on line (0-based)
    int acol = col;
    for (int i=0; i<line.length(); i++) {
      char c = line.charAt(i);
      if (c != '\t') {
        adjusted.append(c);
        x++;
      } else {
        adjusted.append(' ');		// Tab always non-zero skip
        x++;
        while (x%TABSTOP != 0) {	// Skip to next tab stop
          adjusted.append(' ');
          x++;

          if (i<col)
            acol++;
        }
      }
    }
    line = adjusted.toString();
    col = acol;


    // Handle case where file has been modified since it was last read:
    if (col>=line.length()) {
      System.out.println("(line text no longer available; see column "
                         + col + ")");
      return;
    }


    String noclipLine = line;
    int noclipCol = col;
    final int LINELENGTH = 80;	// HACK? !!!!

    /*
     * Clip the start of line if column would otherwise be too far to
     * the right:
     */
    if (col>LINELENGTH-15) {
      int clip = col-(LINELENGTH/2 - 5); 
      //@ assert col - clip + 5 < LINELENGTH - 15;
      col = col + 5 - clip; // This 5 is the length of " ... "
      line = " ... " + line.substring(clip);
    }

    /*
     * Clip the end of line if it would go past the edge of the screen:
     */
    if (line.length()>LINELENGTH-10) {
      line = line.substring(0, LINELENGTH-10) + " ...";
    } else if (policy != null &&
               !policy.containsEndOfConstruct(noclipLine, noclipCol)) {
      line += " ...";
    }

    System.out.println(line);

    // Display a ^ where the col. # is:
    for (int o=0; o<col; o++)
      System.out.print(" ");
    System.out.println("^");
  }

  /** Prints to System.out the given String (if not null)
      and a current stack trace,
      to be used for debugging with print statements.
  */
  //@ public normal_behavior
  //@ modifies System.out.output;
  //@ ensures true;
  static public void dump(String s) {
    if (s != null) System.out.println(s);
    (new Exception()).printStackTrace();
  }

  public static interface Reporter {
    /**
     * Unified interface for reporting information - all messages to the
     * user go through this method.
     * @param severity - the severity of the condition: 0 for information
     *	1 for warnings, 2 for errors
     * @param loc - the location (as in @loc(javafe.util.Location))
     *	referred to by the message; Location.NULL if the message
     *	does not refer to any location in particular
     * @param length - the length of the section of the line that should
     *	be high-lighted; -1 if the length is not known.
     * @param message - the text message to be conveyed to the user
     */
    void report(int severity, int loc, int length, String message);

    /** This method reports the location of an associated bit of
     *  information (e.g. the location of a referenced declaration)
     *  that goes with the most recent call of 'report'.
     * @param loc The Location of theassociated information
     */
    void reportAssociatedInfo(int loc);
    void reportAssociatedInfo2(int loc, ClipPolicy cp);
  }

  static private Reporter reporter = new StandardReporter();

  /** Returns the current output reporter.
      @return the current reporter
  */
  static public Reporter getReporter() {
    return reporter;
  }

  /**
   * Sets the reporter to be used to convey information to the user;
   * returns the previous value of the reporter.
   *
   * @param r The new value of the single reporter object.
   * @return  The previous value of the reporter object.
   */
  public static Reporter setReporter(Reporter r) {
    Reporter rr = reporter;
    reporter = r;
    return rr;
  }

  public static class StandardReporter implements Reporter {
    public void report(int severity, int loc, 
                       int length, String message) {
      if (loc == Location.NULL) {
        System.out.println(message);

      } else {
        System.out.print(Location.toFileName(loc) + ":");
        if (!Location.isWholeFileLoc(loc))
          System.out.print(Location.toLineNumber(loc) + ":");

        // Display the type of the information & the information:
        System.out.println(" " + message);

        // Display which column # visually if available:
        if (!Location.isWholeFileLoc(loc))
          displayColumn(loc);
        else if (message.length() == 0) {
          System.out.println("                ( line not available )");
          System.out.println("");
        }
      }
    }

    public void reportAssociatedInfo(int loc) {
      if (loc != Location.NULL)
        report(0, loc, -1, "Associated declaration: ");
    }

    // Alternate syntax for associated declarations - they
    // should be unified, but that involves fixing all of the tests - FIXME
    public void reportAssociatedInfo2(int loc, ClipPolicy cp) {
      System.out.println("Associated declaration is "
                         + Location.toString(loc) + ":");
      if (!Location.isWholeFileLoc(loc)) {
        ErrorSet.displayColumn(loc, cp);
      }
    }


  }
}
