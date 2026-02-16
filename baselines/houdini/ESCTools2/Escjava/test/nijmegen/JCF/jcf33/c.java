/*
 Adaptation of Bergstra & Loots's Java Class Family number 33:
  
 Results are written to boolean variables `resulti' instead of
 being printed.
  
 Tuesday 15 June 99 19:26:06 bart@frustratie

 Annotations added for ESC/Java2.
 Sunday 11 January 04 21:35:11 bart@depri

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf33;

class c {

    static boolean result1, result2, result3, result4;  

    /*@ normal_behavior
     @   requires true;
     @ modifiable result1, result2, result3, result4, c1.a;
     @    ensures (result1 == \old(c1.a)) &&
     @            (result2 == true) &&
     @            (result3 == \old(c1.e)) &&
     @            (result4 == \old(c1.d));
     @*/    
    public static void m() {
        c1 x = new c1();
        result1 = x.a;
        x.a = true;
        result2 = x.a;
        result3 = c1.e;
        result4 = c1.d;
    }

}

class c1 {
  
    static boolean d = true;

    static boolean b = e;

    static boolean a = b;

    static boolean e = !d;

}
