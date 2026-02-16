/*
 Adaptation of Bergstra & Loots's Java Class Family number 34:
  
 Results are written to boolean variables `resulti' instead of
 being printed.
  
 Tuesday 15 June 99 19:26:06 bart@frustratie


 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf34;

public class c {

    static boolean result1, result2, result3, result4;  

    public static void m() {
        result1 = c1.a;
        result2 = c1.b;
        result3 = c1.x.a;
        result4 = c1.x.b;
    }

}

class c1 {
  
    static c1 x = new c1();

    static boolean b = !x.a;

    static boolean a = false; // putting true here makes all
    // resulti variables true.

}


class Test {

    public static void main(String[] args) {
        c t = new c();
        t.m();
        System.out.println(t.result1);
        System.out.println(t.result2);
        System.out.println(t.result3);
        System.out.println(t.result4);
    }

}
