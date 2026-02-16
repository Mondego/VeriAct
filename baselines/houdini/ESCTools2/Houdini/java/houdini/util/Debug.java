/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini.util;

import java.io.*;
import java.util.*; 


/**
 * class providing a global debugger flag and other debugging tools.
 * this class reads information from debug.properties.
 */
public class Debug {
    
    /**
     * global flag to turn debug code on and off.
     */
    static public boolean debug = true;
    
    static {
        Properties p = new Properties();
        try {
            File f = Utility.findFileOnClasspath("debug.properties");
            if (f == null) {               
                debug = false;
            } else {
                p.load(new FileInputStream(f));
                String s = p.getProperty("debug.debug", "false");
                debug = Boolean.valueOf(s).booleanValue();
            }
        } catch (Exception e) {
            Assert.notify(e);
        }
    }
    
    /**
     * return a string containing the current stack.
     */
    static public String getStackDump() {
        Writer w = new StringWriter();
        PrintWriter pw = new PrintWriter(w);
        Throwable t = new Throwable();
        t.fillInStackTrace();
        t.printStackTrace(pw);
        String stack = w.toString();
        return stack;
    }    
    
    /**
     * Enforce stdout/err logging.
     */
    static {
        try {
            if (Debug.debug) {
                PrintStream log = new PrintStream(new FileOutputStream("debug.log"), true);
                System.setOut(new SplitPrintStream(System.out, log));
                System.setErr(new SplitPrintStream(System.err, log));
            }
        } catch (Exception e) {
            Assert.fail(e);
        }
    }
}

