package escjava.vcGeneration;

import java.io.PrintStream;


/**
 * The class to print info message, warning messages and error messages.
 */
public class TDisplay {
	/** tell whether or not errors should be printed */
    private static boolean bErr;
	/** tell whether or not warnings should be printed */
    private static boolean bWarn;
	/** tell whether or not infos should be printed */
    private static boolean bInfo;
    /** the stream to use to print errors */
    private static PrintStream sErr = System.err;
    /** the stream to use to print warnings */
    private static PrintStream sWrn = System.err;
    /** the stream to use to print infos */
    private static PrintStream sNfo = System.err;
    /** the error prompt */
    static/*@ non_null @*/String errPrompt;
    /** the warning prompt */
    static/*@ non_null @*/String warnPrompt;
    
    /**
     * Initialize the TDisplay class with it options, especially to
     * tell if, how and what should be printed.
     * @param err to tell whether or not to print the error messages
     * @param warn to tell whether or not to print the warning messages
     * @param info to tell whether or not to print the info messages
     * @param colors to trigger the color mode
     */
    static public void init(boolean err, boolean warn, boolean info, boolean colors) {

        TDisplay.bErr = err;
        TDisplay.bWarn = warn;
        TDisplay.bInfo = info;

        if (!colors) {
            TDisplay.errPrompt = "Err ";
            TDisplay.warnPrompt = "Warning ";
        } else {
            TDisplay.errPrompt = "\033[31m>\033[0;m ";
            TDisplay.warnPrompt = "\033[33m>\033[0;m ";
        }

    }

    /**
     * Print an error message, with the program point where the
     * method was called.
     * @param err the error message
     */
    public static void err(String err) {
        if (TDisplay.bErr) {
        	StackTraceElement [] ste = new Exception().getStackTrace();	
            sErr.println(errPrompt + ste[1]);
            sErr.println("  " + err);
        }
    }    
    
    /**
     * Print an warning message, with the program point where the
     * method was called.
     * @param err the warning message
     */
    public static void warn(String warn) {

        if (TDisplay.bWarn) {
        	StackTraceElement [] ste = new Exception().getStackTrace();	
        	sWrn.println(warnPrompt + ste[1]);
            sWrn.println("  " + warn);
        }
    }
    
    /**
     * Print an info message, with the program point where the
     * method was called.
     * @param info the info message
     */
    public static void info(String info) {
        if (TDisplay.bInfo) {
        	StackTraceElement [] ste = new Exception().getStackTrace();	
            sNfo.println("[" + ste[1] + "]");
            sNfo.println("[" + info + "]");
        }
    }
    
    /**
     * @deprecated use {@link #err(String)} instead
     */
    public static void err(/*@ non_null @*/Object o, /*@ non_null @*/String method, String err) {

        if (TDisplay.bErr) {
            sErr.println(errPrompt + o.getClass().getName() + "."
                    + method);
            sErr.println("  " + err);
        }
    }

    /**
     * @deprecated use {@link #warn(String)} instead
     */
    public static void warn(/*@ non_null @*/Object o, /*@ non_null @*/String method, String warn) {

        if (TDisplay.bWarn) {
        	sWrn.println(warnPrompt + o.getClass().getName() + "."
                    + method);
            sWrn.println("  " + warn);
        }
    }

    /**
     * @deprecated use {@link #info(String)} instead
     */
    public static void info(/*@ non_null @*/Object o, /*@ non_null @*/String method, String info) {

        if (TDisplay.bInfo) {
            sNfo.println("[" + o.getClass().getName() + "." + method
                    + "]");
            sNfo.println("[" + info + "]");
        }

    }

    /**
     * @deprecated use {@link #err(String)} instead
     */
    public static void err(/*@ non_null @*/String o, /*@ non_null @*/String method, String err) {

        if (TDisplay.bErr) {
            sErr.println(errPrompt + o + "." + method);
            sErr.println("  " + err);
        }

    }

    /**
     * @deprecated use {@link #warn(String)} instead
     */
    public static void warn(/*@ non_null @*/String o, /*@ non_null @*/String method, String warn) {

        if (TDisplay.bWarn) {
            sWrn.println(warnPrompt + o + "." + method);
            sWrn.println("  " + warn);
        }

    }

    /**
     * @deprecated use {@link #info(String)} instead
     */
    public static void info(/*@ non_null @*/String o, /*@ non_null @*/String method, String info) {

        if (TDisplay.bInfo) {
            sNfo.println("[" + o + "." + method + "]");
            sNfo.println("[" + info + "]");
        }

    }

}
