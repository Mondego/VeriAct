/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe;

/**
 * <code>Tool</code> is an abstract class for tools.
 *
 * <p> Tools are command-line applications invoked by calling their
 * static <code>main(String[])</code> method. <p>
 */

public abstract class Tool
{
  /**************************************************
   *    Exit codes                                  *
   **************************************************/

  static public final int okExitCode = 0;
  static public final int badUsageExitCode = 1;
  static public final int errorExitCode = 2;
  static public final int outOfMemoryExitCode = 3;

  /***************************************************
   *                                                 *
   * Generating a usage message:		     *
   *                                                 *
   **************************************************/

  /**
   * Return the non-null name of this tool.  E.g., "ls" or "cp".
   * Used in usage and error messages.
   */
  //@ ensures \result != null;
  public abstract String name();


  /**
   * Print our usage message to <code>System.err</code>.
   */
  public void usage() {
    getOptions().usage(name());
  }

  public void badOptionUsage(/*@non_null*/Exception e) {
    System.err.println(name() + ": " + e.getMessage());
    if (!getOptions().quiet) usage();
  }

  /***************************************************
   *                                                 *
   * Generic option processing:		             *
   *                                                 *
   **************************************************/

  /** 
   * A statically held Options object.  The object is static to
   * facilitate using the options in other classes throughout the
   * program.  All processing and reporting of options is managed by
   * this object.
   */
  static public /*@nullable*/Options options = null;
    
  /***************************************************
   *                                                 *
   * Main processing code:			     *
   *                                                 *
   **************************************************/

  //+@ requires options != null;
  public static /*@non_null*/Options getOptions() {
	return /*+@(non_null)*/options;
  }


/**
   * Start up an instance of this tool using command-line arguments
   * <code>args</code>.
   *
   * <p> <strong>Note</strong>: this code needs to be copied verbatim
   * to each subclass of <code>Tool</code> except with the name of the
   * actual subclass inserted after the new operator and the comment
   * characters (//) removed.
   *
   * <p> (This needs to be done because static methods cannot be
   * inherited.) <p>
   */
  //@ requires \nonnullelements(args);
  public static void main(/*@non_null*/String[/*#@non_null*/] args) {
    // Tool t = new Tool();
    // int result = t.run(args);
    // if (result != 0) System.exit(result);
  }


  /**
   * A tool's main entry point; <code>args</code> are the
   * command-line arguments we have been invoked with.
   *
   * @return the exit code (0 = success, >0 is a failure)
   */
  //@ requires \nonnullelements(args);
  public abstract int run(/*@non_null*/String[/*#@non_null*/] args);

  /**
   * Compute the time used from a start time to now, then return it in
   * a user readable form.
   */
  public static /*@non_null*/String timeUsed(long startTime) {
    if (getOptions().testMode) return "TIME";
    long delta = java.lang.System.currentTimeMillis() - startTime;
     
    return (delta/1000.0) + " s" + " " + spaceUsed();
  }
 
  public static long currentTime() {
    return java.lang.System.currentTimeMillis();
  }
 
  public static String spaceUsed() {
    long used = rt.totalMemory() - rt.freeMemory();
    return used + " bytes";
  }

  private static /*@non_null*/Runtime rt = Runtime.getRuntime();
}
