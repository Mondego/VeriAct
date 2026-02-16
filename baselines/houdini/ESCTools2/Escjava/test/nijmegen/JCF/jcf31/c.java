/*
 Adaptation of Bergstra & Loots's Java Class Family number 31:
  
 Results are written to boolean variables `resulti' instead of
 being printed.
  
 Sunday 13 June 99 12:49:02 bart@frustratie

 Annotations added for ESC/Java2.
 Sunday 11 January 04 21:24:48 bart@depri

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf31;

class c {

    static boolean result1, result2, result3, result4, result5, result6,
            result7, result8, result9;


    /*@ normal_behavior
     @   requires true;
     @ modifiable result1, result2, result3, result4, 
     @            result5, result6, result7, result8, 
     @            result9, c1.d, c1.e;
     @    ensures (result1 == false) &&
     @            (result2 == true) &&
     @            (result3 == false) &&
     @            (result4 == true) &&
     @            (result5 == false) &&
     @            (result6 == false) &&
     @            (result7 == false) &&
     @            (result8 == true) &&
     @            (result9 == false);
     @*/    
    public static void m() {
        c1 x = new c1();
        c1.d = false;
        c1.e = true;
        x.b = false;
        x.c = true;
        result1 = c1.d;
        result2 = c1.e;
        result3 = x.b;
        result4 = x.c;
        x.b = x.c;
        c1 y = new c1();
        y.b = y.c = x.c = false;
        result5 = y.b;
        result6 = y.c;
        result7 = x.c;
        result8 = x.b;
        result9 = y.d;
    }

}

class c1 {
  
    boolean b = true, c = false;

    static boolean d = true, e = false;

}


