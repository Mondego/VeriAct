/*
 Adaptation of Bergstra & Loots's Java Class Family number 41:
  
 Results are not printed, but stored in a special static
 integer array.
  
 Aim: illustrate the left-most inner-most evaluation.
  
 Thursday 3 June 99 9:34:52 bart@frustratie

 Annotations added for ESC/Java2
 (but unsuccesful because of static initialization)
 Sunday 11 January 04 22:12:42 bart@depri

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf41;

class c {

    public static void m() {
        c1.f(c4.x.g2().g2(), c4.x.g1()).y.m();
        c2.f(c4.x.g1().g1(), c4.x.g2()).y.m();
    }

}

class c0 {

    /*@ normal_behavior
     @   requires R.pos < R.r.length;
     @ modifiable R.pos, R.r[R.pos]; 
     @    ensures R.pos == \old(R.pos) + 1 &&
     @            R.r[\old(R.pos)] == 1;
     @*/  
    public void m() {
        R.store(1);
    }

}

class c1 {

    /*@ normal_behavior
     @   requires R.pos < R.r.length;
     @ modifiable R.pos, R.r[R.pos]; 
     @    ensures R.pos == \old(R.pos) + 1 &&
     @            R.r[\old(R.pos)] == 3 &&
     @            (\result == x);
     @*/    
    static c3 f(c3 x, c3 y) {
        R.store(3);
        return x;
    }

}

class c2 {

    /*@ normal_behavior
     @   requires R.pos < R.r.length;
     @ modifiable R.pos, R.r[R.pos]; 
     @    ensures R.pos == \old(R.pos) + 1 &&
     @            R.r[\old(R.pos)] == 2 &&
     @            (\result == x);
     @*/    
    static c3 f(c3 x, c3 y) {
        R.store(2);
        return x;
    }

}

class c3 { 

    c0 y = new c0();

    /*@ normal_behavior
     @   requires R.pos < R.r.length;
     @ modifiable R.pos, R.r[R.pos]; 
     @    ensures R.pos == \old(R.pos) + 1 &&
     @            R.r[\old(R.pos)] == 4 &&
     @            (\result == this);
     @*/    
    c3 g1() {
        R.store(4);
        return this;
    }

    /*@ normal_behavior
     @   requires R.pos < R.r.length;
     @ modifiable R.pos, R.r[R.pos]; 
     @    ensures R.pos == \old(R.pos) + 1 &&
     @            R.r[\old(R.pos)] == 5 &&
     @            (\result == this);
     @*/    
    c3 g2() {
        R.store(5);
        return this;
    }

}

class c4 { 
    static c3 x = new c3(); 
}

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


