/* Copyright 2000, 2001, Compaq Computer Corporation */

package houdini;

import escjava.prover.*;

import java.io.*;
import java.util.Enumeration;

import javafe.util.*;


public class SimplifyTest {

    /**
     ** A simple test routine
     **/
    public static void main(String[] args) throws IOException {
	Info.on = true;		// Turn on verbose mode

	Simplify S = new Simplify();

        DataInputStream in = new DataInputStream(new FileInputStream(args[0]));
        byte b[] = new byte[in.available()];
        in.readFully(b);
        String s = new String(b);
        try {
	    exhaust(S.prove(s));
	} catch (Exception e) {
	    System.out.println("dead");
	    try {
	        S.close();
	    } catch (Exception exc) {
	    }
	    System.out.println("closed");
	}
	S = new Simplify();
        exhaust(S.prove(s));
    }


    /**
     ** Force an Enumeration to compute all of its elements
     **/
    static void exhaust(Enumeration n) {
	while (n.hasMoreElements())
	    n.nextElement();
    }
}
