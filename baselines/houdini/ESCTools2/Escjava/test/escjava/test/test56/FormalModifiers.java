/**
 ** Test that we can put escjava modifiers after formal parameters
 **/

class FormalModifiers {
    //@ pure
    void foo (String x /*@non_null*/,
	      Object z[] /*@non_null*/,
	      int[] q /*@non_null*/ /*@non_null*/) {}
    //@ pure
    void bar (/*@non_null*/ String a /*@non_null*/) {}
    //@ pure
    void b1 (int b) {}
    //@ pure
    void b2 (/*@non_null*/ String b) {}
}
