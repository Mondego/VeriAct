/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini;

import escjava.prover.*;
import java.io.*;

/**
 * Reads in the pickled SExp form and prints it.
 */
class SExpPrinter {

  public static void main(String st[])  throws Exception {
    DataInputStream in = new DataInputStream(new FileInputStream(st[0]));
    SExp t = SExpOptimizer.readFromFile(in);
    System.out.println(t);
  }
  
}
	
