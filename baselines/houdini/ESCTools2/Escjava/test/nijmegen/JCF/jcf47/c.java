/*
 Adaptation of Bergstra & Loots's Java Class Family number 47:
  
 Result will be returned as integer, and not printed.
  
 Wednesday 16 June 99 0:25:17 bart@frustratie

 Annotations added for ESC/Java2;
 Monday 12 January 04 21:56:12 bart@depri

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf47;

class c {

    /*@ normal_behavior
     @   requires true;
     @ modifiable \nothing;
     @    ensures \result == 100;
     @*/    
    public static int m() {
        return c1.z.y(c2.x).m();
    }

}

class c1 {
  
    /*@
     @ invariant z != null;
     @*/

    static c1 z = new c1();

    /*@ normal_behavior
     @   requires true;
     @ modifiable \nothing;
     @    ensures \result != null;
     @*/  
    c1 y (c1 z) { return new c1(); }

    /*@ normal_behavior
     @   requires true;
     @ modifiable \nothing;
     @    ensures \result == 100;
     @*/    
    public int m() { return 100; }

}

class c2 {

    static c1 x = new c1();

}
