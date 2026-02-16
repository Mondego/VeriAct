/*
 Adaptation of Bergstra & Loots's Java Class Family number 55:
  
 Results are written to boolean variables `resulti' in a special 
 Result class, instead of being printed.
  
 The identifiers `_x' are replace by `x', because `_x' gives
 parser problems in PVS.
  
 Thursday 17 June 99 17:09:26 bart@cicero.cs.kun.nl

 Annotations added for ESC/Java2;
 Monday 12 January 04 21:56:12 bart@depri

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf55;

class c {

    static Result m() {
	return c1.m1();
    }

}

class c1 {

    /*@ normal_behavior
     @   requires true;
     @ modifiable X.e, X.f;  
     @    ensures \result != null;
     @    ensures !\result.result1;
     @    ensures \result.result2;
     @    ensures \result.result3 == \old(X.f);
     @    ensures \result.result4;
     @    ensures \result.result5;
     @    ensures \result.result6;
     @    ensures !\result.result7;
     @    ensures !\result.result8;
     @*/
    static Result m1() {
	Result r = new Result();
	X x = new X();
	x.b = false;
	r.result1 = x.b;
	r.result2 = x.c;
	x.c = true;
	r.result3 = X.f;
	X z = new X(true, false);   // z.b=true, z.c=false, z.d=false
	//@ assert z.b;
	//@ assert !z.c;
	r.result4 = !z.c;           // r.result4=true
	X u = new X(z);             // u.b=z.c=false, u.c=z.b=true
	//@ assert !z.c;
	//@ assert z.b;
	//@ assert u.c;
	//@ assert !u.b;
	r.result5 = !u.b;           // true
	//@ assert r.result5;
	x = z.next(u);              // x.b=u.b=false, z.c=false, x.c=u.c=true
	//@ assert u.c;
	//@ assert x.c;
	r.result6 = x.c;            // true
	r.result7 = x.b;
	r.result8 = X.f;
 	return r;
    }

    // Added by Joe on 19 Jan 2004 to determine if the bug wrt ref
    // formal parameter fields and modifies clauses is restricted to
    // constructors or not.
    /*@ normal_behavior
     @   requires true;
     @ modifiable X.e, X.f;  
     @    ensures \result != null;
     @    ensures !\result.result1;
     @    ensures \result.result2;
     @    ensures \result.result3 == \old(X.f);
     @    ensures \result.result4;
     @    ensures \result.result5;
     @    ensures \result.result6;
     @    ensures !\result.result7;
     @    ensures !\result.result8;
     @*/
    static Result m2() {
	Result r = new Result();
	X x = new X();
	x.b = false;
	r.result1 = x.b;
	r.result2 = x.c;
	x.c = true;
	r.result3 = X.f;
	X z = new X(true, false);
	r.result4 = !z.c;
	X u = new X();
        u.methodX(z);
	r.result5 = !u.b;
	x = z.next(u);
	r.result6 = x.c;
	r.result7 = x.b;
	r.result8 = X.f;
 	return r;
    }

}

class X {

    public boolean b, c = true;
    
    private boolean d = false;

    static private /*@ spec_public */ boolean e = false;

    static public boolean f = true;

    /*@ normal_behavior
     @   requires true;
     @ modifiable e, d;  
     @    ensures c && (e == !\old(e)) && (d == !\old(e)) && (f == \old(f));
     @*/   
    X() {
	e = !e;
	d = e;
    }

    /*@ normal_behavior
     @   requires true;
     @ modifiable b, c, d;  
     @    ensures (b == b1) && (c == c1) && (d == (b1 && c1)) && 
     @            (f == \old(f));
     @*/ 
    X(boolean b1, boolean c1) {
	b = b1;
	c = c1;
	d = b1 && c1;
    }

    /*@ normal_behavior
     @   requires x != null && x != this;
     @ modifiable b, c, d;  
     @    ensures (b == x.c) && (c == x.b) && (d == e);

     // @bug bart,joe - the next two predicates must be stated to
     //                 verify the postcondition of m1(), but given
     //                 the modifiable clause above, shouldn't they be
     //                 implied?  In other words, while all ref formal
     //                 parameters are implicitly part of the modifies
     //                 clause of a method, shouldn't modifying all
     //                 fields of such parameters be disallowed
     //                 implicitly?

     //    ensures (x.b == \old(x.b)) && (x.c == \old(x.c));
     //    ensures x == \old(x);
     @*/ 
    X(X x) {
	b = x.c;
	c = x.b;
	d = e;
    }

    // Added by Joe on 19 Jan 2004 to determine if the bug wrt ref
    // formal parameter fields and modifies clauses is restricted to
    // constructors or not.
    /*@ normal_behavior
      @   modifiable b, c, d;
      @    ensures (b == x.c) && (c == x.b) && (d == e);
      @*/
    void methodX(/*@ non_null @*/ X x) {
	b = x.c;
	c = x.b;
	d = e;
    }

    /*@ normal_behavior
     @   requires x != null && x != this;
     @ modifiable c, x.b, x.d, e, f;  
     @    ensures (\result == x) &&
     @            !x.b && !c && !x.d && !e && !f;
     @*/ 
    X next(X x) {
	X y = new X(true, true);
	y = x;
	y.b = this.c = y.d = y.e = y.f = false;
	return y;
    }	

}

class Result {

    boolean result1, result2, result3, result4, 
            result5, result6, result7, result8;

}
