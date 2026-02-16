/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini.util;

import java.io.*;

/**
 * Various routines for testing assertions and reporting errors.
 */
public class Assert {
    
    /**
     * If passed false, this method well report an assertion failure
     * and exit the program.
     */
    //@ ensures b;
    static public void notFalse(boolean b) {
        if (!b) {
            System.err.println("assertion failed");
            System.err.println(Debug.getStackDump());
            houdini.util.ShutDown.exit(0);
        }
    }

    /**
     * If passed false, this method well report an assertion failure,
     * print the string passed in, and exit the program.
     */
    //@ ensures b;
    static public void notFalse(boolean b, String s) {
        if (!b) {
            System.err.println("assertion failed: " + s);
            System.err.println(Debug.getStackDump());
            houdini.util.ShutDown.exit(0);
        }
    }
    
    /**
     * Prints the exception e, dumps the stack where e occurred, and exits.
     */
    //@ ensures false;
    static public void fail(Throwable e) {
        System.err.println("fail: " + e);
        e.printStackTrace();
        houdini.util.ShutDown.exit(0);
    }
    
    /**
     * Prints the exception e and message s, dumps the stack where e 
     * occurred, and exits.
     */
    //@ ensures false;
    static public void fail(Throwable e, String s) {
        System.err.println("fail: " + s);
        fail(e);
    }    
    
   /**
     * Prints the message s, dumps the current stack, and exits.
     */
    //@ ensures false;
    static public void fail(String s) {
        fail(new Throwable(), s);
    }    

    /**
     * Same as fail, but does not exit.
     */
    static public void notify(String s) {
        System.err.println(s);
    }
    
    static public void notify(Throwable e) {
        System.err.println("notify: " + e);        
        e.printStackTrace();
    }    
    
    static public void notify(Throwable e, String s) {
        System.err.println("notify: " + s);
        notify(e);
    }   
    
}
