/*
 Adaptation of Bergstra & Loots's Java Class Family number 15:
  
 Results are written to boolean variables `resulti' instead of
 being printed.
  
 This example involves mutual recursion.
  
 Sunday 13 June 99 12:49:02 bart@frustratie

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf15;

class c {
    boolean result1, result2;
  
    void m() {
        result1 = d1.b1;
        result2 = d2.b2;
    }
  
}

final class d1 {
    static boolean b1 = !d2.b2;
}

final class d2 {
    static boolean b2 = !d1.b1;
}

