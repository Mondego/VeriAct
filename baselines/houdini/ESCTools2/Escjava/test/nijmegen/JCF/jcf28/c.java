/*
 Adaptation of Bergstra & Loots's Java Class Family number 28:
 static fields and methods are replaced by
 instance fields and methods; results are not
 printed, but written to boolean instance fields `resulti'.

 Aim: to show that side-effects result in difference
 between disjunction and disjunction in Java and in 
 standard logic.

 Wednesday 2 June 99 23:31:54 bart@frustratie

 Annotations added for ESC/Java2.
 Saturday 10 January 04 12:56:06 bart@depri

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf28;

class c {

    static boolean result1, result2;

    static boolean b = true;

    /*@ normal_behavior
     @   requires true;
     @ modifiable b;
     @    ensures (b == !\old(b)) && (\result == b);
     @*/    
    static boolean f() { 
        b = !b;
        return b;
    }

    /*@ normal_behavior
     @   requires true;
     @ modifiable b, result1, result2;
     @    ensures b && result2 && (result1 == !\old(b));
     @*/    
    static void m() {
        result1 = f() || !f();  // yields false if \old(b)
        result2 = !f() && f();  // yields true
    }

}
