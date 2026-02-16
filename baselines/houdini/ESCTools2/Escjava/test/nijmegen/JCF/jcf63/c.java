/*
 Adaptation of Bergstra & Loots's Java Class Family number 63:

 Results are written to boolean variables `resulti' instead of
 being printed.

 Thursday 17 June 99 21:27:54 bart@frustratie

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf63;

class c extends c1 { }


class c1 extends c2 {

    static boolean result1, result2, result3, result4, result5, result6;
  
    static void m() {
        result1 = a;
        result2 = b;
        result3 = c;
        result4 = d;
        result5 = e;
        result6 = !f;
    }

}


class c2 extends c3 {
  
    static boolean b = true;

    static boolean c = false;

    static boolean d = !c3.d;

}


class c3 {

    static boolean a = false,
            b = false,
            c = false,
            d = true,
            e = true,
            f = true;

}
