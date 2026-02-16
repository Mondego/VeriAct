/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini.util;

import java.util.*;
import java.io.*;
import java.lang.System;
import java.lang.Runtime;

/**
 * This class provide services for gracefully shutting down a program.
 * Depending upon finalizers to deallocate resources correctly is 
 * problematic sense as the program shuts down, there's no guarantee
 * that they will be called while the program is in a reasonable state.
 *
 * This class provides a way to run code while the system exits but
 * before objects would be finalized during shutdown.
 */
public class ShutDown {
    /** list of runnable objects */
    static private Vector v = new Vector();
    
    /** add runnable object r to the list of objects to execute prior to shut down */
    static public void runOnExit(Runnable r) {
        v.addElement(r);
    }
    
    /**
     * Replaces {@link System#exit(int)} with a version that executes the previously
     * recorded runnable objects.
     */
    static public void exit(int status) {
        for (int i = 0; i < v.size(); i++) {
            Runnable r = (Runnable)v.elementAt(i);
            r.run();
        }
        System.exit(status);
    }
}
