/*
 Adaptation of Bergstra & Loots's Java Class Family number 68:
  
 The result is returned as boolean, and is not printed.
  
 Monday 21 June 99 22:11:16 bart@frustratie

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf68;

public class c {

    static c1 x = new c1();

    static boolean m() { return x.m(); }

}

class c1 {

    static boolean q = true;

    public boolean m(){ return q; }

}

