/*
 Adaptation of Bergstra & Loots's Java Class Family number 82:
  
 The result is returned as boolean, not printed.
  
 Thursday 24 June 99 10:00:47 bart@cicero.cs.kun.nl

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf82;

public class c {
  
    static c2 x = new c2();

    public static void m(){x.m();}
}  

class c1 {

    static boolean q = true;

    c1() {
        q = false;
        Result.r[Result.pos] = true;
        Result.pos++;
    }
}

class c2 extends c1 {

    public void m(){
        Result.r[Result.pos] = c1.q;
        Result.pos++;
    }
}

class Result {
  
    static boolean[] r = new boolean[2];
    static int pos = 0;
  
}
