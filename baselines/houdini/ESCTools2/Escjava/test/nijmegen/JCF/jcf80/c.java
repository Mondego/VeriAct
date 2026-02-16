/*
 Adaptation of Bergstra & Loots's Java Class Family number 80:
  
 The result is returned as boolean, not printed.
  
 Thursday 24 June 99 10:00:47 bart@cicero.cs.kun.nl

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf80;

public class c {

    static c3 x = new c3();
    public static boolean m(){ return x.m();}
}  

class c1 {
    private boolean q = true;
}

class c2 extends c1 {
    boolean q;
}

class c3 extends c2 {
    public boolean m(){ return q;}
}
