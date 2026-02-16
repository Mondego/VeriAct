package decsrc.util;

import java.io.PrintWriter;
import java.io.StringWriter;
import java.io.OutputStreamWriter;

/** eXtra utility routines. */
public abstract class X {
  /** Debugging output is printed iff <code>debug_mode</code> is true. */
  public static boolean debug_mode = false;

  /** Formatter that sends output to the standard output stream. */
  public static Formatter out =
    new Formatter(new OutputStreamWriter(System.out));

  /** Formatter that sends output to the standard error stream. */
  public static Formatter err =
    new Formatter(new OutputStreamWriter(System.err));

  /** Raise a failure exception. */
  public static void fail() {
    throw new RuntimeException("Failure");
  }

  /** Raise a failure exception. */
  public static void fail(String s) {
    throw new RuntimeException("Failure: " + s);
  }

  /** Raise NotImplementedException. */
  public static void notImplemented(String s) {
    throw new NotImplementedException("Not implemented: " + s);
  }

  /** Raise an assertion-failure exception iff <code>c</code> is false. */
  public static void assertTrue(boolean c) {
    if (! c) throw new RuntimeException("Assertion failure");
  }

  /** Raise an assertion-failure exception iff <code>c</code> is false. */
  public static void assertTrue(boolean c, String s) {
    if (! c) throw new RuntimeException("Assertion failure: " + s);
  }

    /** Add '\'s so that embedded quotes are escaped */
    public static String escape(String s) {
	if (s.indexOf('"') == -1 &&
	    s.indexOf('\n') == -1 &&
	    s.indexOf('\\') == -1) {
	    return s;
	}
	StringBuffer buf = new StringBuffer();
	for (int i = 0; i < s.length(); i++) {
	    char c = s.charAt(i);
	    if (c == '"' || c == '\n' || c == '\\') {
		buf.append('\\');
	    }
	    buf.append(c);
	}
	return buf.toString();
  }

  /** Print <code>s</code> to standard output iff <code>debug_mode</code>
      is true. */
  public static void debug(String s) {
    if (debug_mode) {
      System.out.print("DBG: ");
      System.out.println(s);
    }
  }

  /** Return stack trace in a string. */
  public static String getStackTrace(Exception e) {
    StringWriter sw = new StringWriter();
    PrintWriter o = new PrintWriter(sw);
    e.printStackTrace(o);
    o.flush();
    return sw.toString();
  }
}

