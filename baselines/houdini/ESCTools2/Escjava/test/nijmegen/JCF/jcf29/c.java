/*
 Adaptation of Bergstra & Loots's Java Class Family number 29:
 Results are not printed, but written to static boolean fields `resulti'.
  
 Aim: to show the flexibility of notation offered by Java.
  
 Thursday 3 June 99 9:34:52 bart@frustratie
  
 Annotations added for ESC/Java2.
 Sunday 11 January 04 21:02:09 bart@depri

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf29;

class c {
  
    static boolean result1, result2, result3, result4, result5, result6, 
            result7, result8, result9, result10, result11, result12;
  
    static boolean c = false;
  
    /*@ normal_behavior
     @   requires true;
     @ modifiable \nothing;
     @    ensures \result == true;
     @*/    
    static boolean b() {
        return true;
    }
  
    /*@ normal_behavior
     @   requires true;
     @ modifiable \nothing;
     @    ensures \result == b;
     @*/    
    static boolean b(boolean b) {
        return b;
    }
  
    /*@ normal_behavior
     @   requires true;
     @ modifiable result1, result2, result3, result4, 
     @            result5, result6, result7, result8, 
     @            result9, result10, result11, result12,
     @            b.b, b.c; 
     @    ensures (result1 == \old(c)) &&
     @            (result2 == true) &&
     @            (result3 == false) &&
     @            (result4 == true) &&
     @            (result5 == \old(b.b)) &&
     @            (result6 == \old(b.c)) &&
     @            (result7 == \old(b.b)) &&
     @            (result8 == true) &&
     @            (result9 == false) &&
     @            (result10 == !\old(b.b)) &&
     @            (result11 == !\old(c)) &&
     @            (result12 == !\old(c));
     @*/    
    static void m() {
        result1 = c;
        result2 = b();
        result3 = b(false);
        result4 = b(true);
        result5 = b.b;
        result6 = b.c;
        result7 = b.b();
        result8 = b.b(false);
        result9 = b.b(b());
        result10 = b.b(b.b);
        result11 = b(!c);
        result12 = !c;
    }

}

class b {

    static boolean b = true;

    static boolean c = true;

    /*@ normal_behavior
     @   requires true;
     @ modifiable \nothing;
     @    ensures \result == b;
     @*/   
    static boolean b() {
        return b;
    }

    /*@ normal_behavior
     @   requires true;
     @ modifiable \nothing;
     @    ensures \result == !b;
     @*/   
    static boolean b(boolean b) {
        return !b;
    }

}
