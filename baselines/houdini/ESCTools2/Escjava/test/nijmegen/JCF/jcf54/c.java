/*
 Adaptation of Bergstra & Loots's Java Class Family number 54:
  
 Results are written to boolean variables `resulti' instead of
 being printed.
  
 The identifier `_u' is replace by `v', because `_u' gives
 parser problems in PVS.
  
 Thursday 17 June 99 10:40:47 bart@frustratie

 No annotations added for ESC/Java2, because all interesting
 properties that are needed for the verification must be
 included in the invariant.
 Monday 12 January 04 21:56:51 bart@depri

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf54;

class c {

    static boolean result1, result2, result3, result4, result5, result6, result7;

    static c1 x1 = new c1();

    static c1 x2 = x1;

    static c1 x3 = x2;

    static c1 x4 = x3;

    static c1 x5 = new c1(x4);

    static c1 x6 = x5.v;

    static boolean b = x6 instanceof c1;


    public static void m() {
        result1 = x1 == x4; /* originally x1.equals(x4) but that's the same
                             in this example */
        result2 = c1.x1.empty;
        result3 = c1.x2.empty;
        result4 = x5.empty;
        result5 = x5.v.empty;
        result6 = b;
        result7 = x6.v.empty;
    }

}

class c1 {
  
    boolean empty;

    c1 v;

    static c1 x1 = new c1();

    static c1 x2 = new c1(x1.empty);

    c1() { 
        empty = true;
    }

    c1(boolean b) {
        empty = false;
    }

    c1(c1 u) {
        empty = false;
        v = u;
    }

}

