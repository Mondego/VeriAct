/*
 Adaptation of Bergstra & Loots's Java Class Family number 43:
 static fields and methods are replaced by
 instance fields and methods; results are not
 printed, but written to an array of integers in a special
 result class.

 Aim: illustrate overloading in combination with innermost evaluation.

 Thursday 3 June 99 9:34:52 bart@frustratie

 Annotations added for ESC/Java2
 Sunday 11 January 04 23:11:23 bart@depri

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf43;

class c {

    /*@ normal_behavior
     @   requires R.pos < R.r.length;
     @ modifiable R.pos, R.r[R.pos]; 
     @    ensures R.pos == \old(R.pos) + 1 &&
     @            R.r[\old(R.pos)] == 1 &&
     @            (\result instanceof c2);
     @*/    
    static c2 g(c1 x) {
        R.store(1);
        return new c2();
    }

    /*@ normal_behavior
     @   requires R.pos < R.r.length;
     @ modifiable R.pos, R.r[R.pos]; 
     @    ensures R.pos == \old(R.pos) + 1 &&
     @            R.r[\old(R.pos)] == 2 &&
     @            (\result instanceof c3);
     @*/ 
    static c3 g(c2 x) {
        R.store(2);
        return new c3();
    }

    /*@ normal_behavior
     @   requires R.pos < R.r.length;
     @ modifiable R.pos, R.r[R.pos]; 
     @    ensures R.pos == \old(R.pos) + 1 &&
     @            R.r[\old(R.pos)] == 3 &&
     @            (\result instanceof c1);
     @*/ 
    static c1 g(c3 x) {
        R.store(3);
        return new c1();
    }

    /*@ normal_behavior
     @   requires R.pos < 83 && R.r.length == 100;
     @ modifiable R.pos, R.r[*]; 
     @    ensures (R.pos == \old(R.pos) + 18) &&
     @            (R.r[\old(R.pos)] == 1) &&     
     @            (R.r[\old(R.pos) + 1] == 2) &&     
     @            (R.r[\old(R.pos) + 2] == 3) &&     
     @            (R.r[\old(R.pos) + 3] == 1) &&     
     @            (R.r[\old(R.pos) + 4] == 2) &&     
     @            (R.r[\old(R.pos) + 5] == 2) &&     
     @            (R.r[\old(R.pos) + 6] == 3) &&     
     @            (R.r[\old(R.pos) + 7] == 3) &&     
     @            (R.r[\old(R.pos) + 8] == 1) &&     
     @            (R.r[\old(R.pos) + 9] == 1) &&     
     @            (R.r[\old(R.pos) + 10] == 2) &&     
     @            (R.r[\old(R.pos) + 11] == 3) &&     
     @            (R.r[\old(R.pos) + 12] == 2) &&     
     @            (R.r[\old(R.pos) + 13] == 3) &&     
     @            (R.r[\old(R.pos) + 14] == 1) &&     
     @            (R.r[\old(R.pos) + 15] == 3) &&     
     @            (R.r[\old(R.pos) + 16] == 1) &&     
     @            (R.r[\old(R.pos) + 17] == 2);
     @*/ 
    static void m() {
        c1 x1 = new c1();
        c2 x2 = new c2();
        c3 x3 = new c3();
        g(x1); g(x2); g(x3);
        g(g(x1)); g(g(x2)); g(g(x3));
        g(g(g(x1))); g(g(g(x2))); g(g(g(x3)));
    }

}

class c1 { }

class c2 { }

class c3 { }

class R {

    /*@ 
     @ invariant r != null && r.length == 100 &&
     @           pos >= 0;
     @*/
  
    static int[] r = new int[100];
    static int pos = 0;

    /*@ normal_behavior
     @   requires pos < r.length;
     @ modifiable pos, r[pos]; // is pos evaluated in pre- or post-state?
     @    ensures pos == \old(pos) + 1 &&
     @            r[\old(pos)] == result;
     @*/  
    static void store(int result) {
        /* Note that the result gets evaluated first */
        r[pos++] = result;
    }
}
