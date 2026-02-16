/*
 Adaptation of Bergstra & Loots's Java Class Family number 17:
  
 Results are written to boolean variables `resulti' instead of
 being printed.
  
 This example involves mutual recursion.
  
 Thursday 8 July 99 9:31:10 bart@irritatie.cs.kun.nl

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf17;

class c {
    static boolean result1, result2;
  
    public static void m() {
        d2.b2 = d2.b2; // (relevant use of d2.b2)
        result1 = d1.b1; // (first output instruction)
        result2 = d2.b2; // (second output instruction)
    }
  
}

final class d1 {
    static boolean b1 = !d2.b2;
}

final class d2 {
    static boolean b2 = !d1.b1;
}

