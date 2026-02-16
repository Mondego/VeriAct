/* Copyright Hewlett-Packard, 2002 */

/**
 ** Test that we can put escjava modifiers after formal parameters
 **/

class FormalModifiers {

    void foo (String x /*@ non_null @*/,
	      Object z[] /*@ non_null @*/,
	      int[] q /*@ non_null @*/ /*@ non_null @*/) {}

    void bar (/*@ non_null @*/ String a /*@ non_null @*/) {}

    void b1 (int b) {}
    void b2 (/*@ non_null @*/ String b) {}
}
