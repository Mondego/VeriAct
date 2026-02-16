/*
 Adaptation of Bergstra & Loots's Java Class Family number 67:
  
 Results are written to integer variables `resulti' instead of
 being printed. The method m below is not static, like in
 the original jcf.
  
 Monday 21 June 99 22:11:16 bart@frustratie

 Annotations added for ESC/Java2;
 Monday 12 January 04 21:56:12 bart@depri

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf67;

class c {

    /*@ 
     @ invariant result != null && result.length == 18 &&
     @           pos >= 0;
     @*/
  
    static int[] result = new int[18];
  
    static int pos = 0;

    /*@ normal_behavior
     @   requires pos < 18;
     @ modifiable pos, result[pos];
     @    ensures pos == \old(pos) + 1 &&
     @            result[\old(pos)] == 1 &&
     @            \result != null &&
     @            \result instanceof c2;
     @*/  
    static c2 g(c1 x) { 
        result[pos] = 1; 
        pos++; 
        return new c2(); 
    }
  
    /*@ normal_behavior
     @   requires pos < 18;
     @ modifiable pos, result[pos];
     @    ensures pos == \old(pos) + 1 &&
     @            result[\old(pos)] == 2 &&
     @            \result != null &&
     @            \result instanceof c3;
     @*/ 
    static c3 g(c2 x) { 
        result[pos] = 2;
        pos++;
        return new c3();
    }
  
    /*@ normal_behavior
     @   requires pos < 18;
     @ modifiable pos, result[pos];
     @    ensures pos == \old(pos) + 1 &&
     @            result[\old(pos)] == 3 &&
     @            \result != null &&
     @            \result instanceof c1;
     @*/ 
    static c1 g(c3 x) { 
        result[pos] = 3;
        pos++;
        return new c1();
    }
  

    /*@ normal_behavior
     @   requires pos == 0;
     @ modifiable pos, result[*];
     @    ensures pos == \old(pos) + 18 &&
     @            result[\old(pos)] == 1 &&
     @            result[\old(pos) + 1] == 2 &&
     @            result[\old(pos) + 2] == 3 &&
     @            result[\old(pos) + 3] == 1 &&
     @            result[\old(pos) + 4] == 2 &&
     @            result[\old(pos) + 5] == 2 &&
     @            result[\old(pos) + 6] == 3 &&
     @            result[\old(pos) + 7] == 3 &&
     @            result[\old(pos) + 8] == 1 &&
     @            result[\old(pos) + 9] == 1 &&
     @            result[\old(pos) + 10] == 2 &&
     @            result[\old(pos) + 11] == 3 &&
     @            result[\old(pos) + 12] == 2 &&
     @            result[\old(pos) + 13] == 3 &&
     @            result[\old(pos) + 14] == 1 &&
     @            result[\old(pos) + 15] == 3 &&
     @            result[\old(pos) + 16] == 1 &&
     @            result[\old(pos) + 17] == 2;
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

class c1 extends c2 { }

class c2 extends c3 { }

class c3 { }
