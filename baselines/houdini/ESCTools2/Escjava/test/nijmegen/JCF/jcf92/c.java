/*
 Adaptation of Bergstra & Loots's Java Class Family number 92:

 The results are stored in static integers, not printed.

 Monday 28 June 99 15:25:34 bart@irritatie.cs.kun.nl

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf92;

class c extends c1 {
    static boolean result1, result2, result3;

    static void m() {
        c1 x = new c1();
        result1 = x.M();
        c2 y1 = new c2();
        result2 = y1.M();
        c2 y2 = x;
        result3 = y2.M();
        //cco.a();
    }
}

class c1 extends c2 {
    boolean M() {return b;}
}

class c2 extends c3 {
    boolean M() {return !b;}
}

class c3 { 
    static boolean b = true;
}
