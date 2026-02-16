/*
 Adaptation of Bergstra & Loots's Java Class Family number 65:
  
 Results are written to static integer variables `resulti' instead of
 being printed.
  
 Monday 21 June 99 22:11:16 bart@frustratie

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf65;

public class c {

    static int result1, result2, result3;  

    static void m() {
        result1 = c1.m();
        result2 = c2.m();
        result3 = c3.m();
    }

}

class c1 extends c2 {

    static int b = 2;

}

class c2 extends c3 {

    static int b = 3;

    static int m() { return b; }

}

class c3 {

    static int m() { return b; }

    static int b = 4;

}
