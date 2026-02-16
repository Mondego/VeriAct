/*
 Adaptation of Bergstra & Loots's Java Class Family number 78:
  
 The result is returned as boolean, and is not printed.
  
 Monday 21 June 99 22:11:16 bart@frustratie

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf78;

class c {

    static Return m() { 
        c3 x3 = new c3();
        return x3.m(); }

}


class c1 {

    public static boolean b = true;
    public boolean c = true;

}


class c2 extends c1 {

    private boolean b = false;
    static boolean c = false;

}


class c3 extends c2 {

    boolean c = false;

    Return m() {
        Return r = new Return();
        c3 y3 = new c3();
        r.return1 = c1.b;
        r.return2 = c;
        r.return3 = c2.c;
        r.return4 = ((c1)y3).c;
        r.return5 = super.c;
        return r;
    }

}

class Return {

    boolean return1, return2, return3, return4, return5;

}

