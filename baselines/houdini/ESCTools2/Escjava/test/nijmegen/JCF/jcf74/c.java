/*
 Adaptation of Bergstra & Loots's Java Class Family number 74:
  
 The result is returned as boolean, and is not printed.
  
 Monday 21 June 99 22:11:16 bart@frustratie

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf74;

class c {

    static c2 x = new c2();

    static Return m() { return x.m(); }

}


class c1 {

    static boolean q = true;

}


class c11 extends c1 { }


class c12 extends c11 {

    static boolean q = false;

}


class c13 extends c12 {

    static boolean q = true;

}


class c14 extends c13 {

    static boolean q = false;

}


class c2 extends c14 {

    public Return m() {
        Return r = new Return();
        r.return1 = q;
        r.return2 = c1.q;
        r.return3 = c11.q;
        r.return4 = c12.q;
        return r;
    }

}

class Return {

    boolean return1, return2, return3, return4;

}

/*
 class JCF74 {

 public static void main(String[] args) {
 Return s = c.m();
 System.out.println(s.return1 + " " + s.return2 + " " +
 s.return3 + " " + s.return4);
 // prints: false true true false
 }

 }*/
