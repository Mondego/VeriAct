/*
 Adaptation of Bergstra & Loots's Java Class Family number 66:
  
 Results are written to integer variables `resulti' instead of
 being printed. The method m below is not static, like in
 the original jcf.
  
 Monday 21 June 99 22:11:16 bart@frustratie

 Annotations added for ESC/Java2;
 Monday 12 January 04 21:56:12 bart@depri

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf66;

class c {

    static int result1, result2, result3, result4, result5, result6;

    /*@ normal_behavior
     @   requires true;
     @ modifiable \nothing;
     @    ensures \result == 1;
     @*/
    static int f(c1 x) { return 1; }

    /*@ normal_behavior
     @   requires true;
     @ modifiable \nothing;
     @    ensures \result == 2;
     @*/
    static int f(c2 x) { return 2; }

    /*@ normal_behavior
     @   requires true;
     @ modifiable \nothing;
     @    ensures \result == 3;
     @*/
    static int f(c3 x) { return 3; }

    /*@ normal_behavior
     @   requires true;
     @ modifiable result1, result2, result3, result4, result5, result6;
     @    ensures result1 == 1 &&
     @            result2 == 2 &&
     @            result3 == 3 &&
     @            result4 == 2 &&
     @            result5 == 3 &&
     @            result6 == 3;
     @*/
    static void m() {
        c1 x1 = new c1();
        c2 x2 = new c2();
        c3 x3 = new c3();
        result1 = f(x1);
        result2 = f(x2);
        result3 = f(x3);
        c2 y2;
        c3 y3, z3;
        y2 = x1;
        y3 = x1;
        z3 = x2;
        result4 = f(y2);
        result5 = f(y3);
        result6 = f(z3);
    }

}

class c1 extends c2 { }

class c2 extends c3 { }

class c3 { }
