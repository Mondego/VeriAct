//#FLAGS: -jmlAssertions
public class JavaAssert
{
    // Currently only testing lightweight specs.

    // The postcondition "ensures false" specifies that the exception
    // *must* be thrown.  This is equivalent to using an
    // exceptional_behavior heavyweight spec case and omitting the
    // ensures clause.  The "requires true" says that the exception
    // will be thrown under all circumstances.

    // Open questions:
    // 1. Is it erroneous, or just nonsensical, to use signals clauses
    //    with RuntimeExceptions or Errors?
    // 2. Should Java asserts be converted into JML asserts?
    // 3. Should Java asserts automatically be annotated with
    //    "nowarn Exception Assert"?
    // 4. Should the default signals clause on a method be
    //    signals (java.lang.RuntimeException) true;
    //    signals (java.lang.Error) true;
    // 

    // Shouldn't this be the default lightweight spec clause?
    //@ requires true;
    //@ ensures true;
    //@ signals (java.lang.Exception) false;
    public void nothing() {
        ;
    }

    //@ requires true;
    //@ ensures false;
    // nowarn Exception Assert;
    public void trivial() {
        assert false;
    }

    //@ requires true;
    //@ ensures false;
    public void constant_boolean_expression() {
        assert true == false;
    }

    //@ requires p >= 0;
    //@ ensures true;
    public void assert_matches_precondition(int p) {
        assert p >= 0;
    }

    //@ requires p >= 0;
    //@ ensures true;
    public void assert_weaker_than_precondition(int p) {
        assert p >= -1;
    }

    // ESC/Java should catch this discrepency and fail on this method.
    //@ requires p >= 0;
    //@ ensures false;
    public void assert_stronger_than_precondition(int p) {
        assert p >= 1;
    }
    
    //@ requires 0 == p;
    //@ requires 10 == q;
    //@ ensures true;
    public void constant_arithmetic(int p, int q) {
        assert p + q == 10;
    }

    // ESC/Java should catch this arithmetic error and fail on this method.
    //@ requires 0 == p;
    //@ requires 10 == q;
    //@ ensures false;
    public void constant_arithmetic_dual(int p, int q) {
        assert p + q == 11;
    }

    // Note that without the upper bounds on p and q this test should
    // fail due to possible overflow.
    //@ requires 0 <= p && p <= 1000;
    //@ requires 0 <= q && q <= 1000;
    //@ ensures 0 <= \result && \result <= 1000*1000;
    public int non_trivial_arithmetic(int p, int q) {
        assert p + q >= 0;
        return p*q;
    }

    // Note that without the upper bounds on p and q this test should
    // fail due to possible overflow.  This test should not pass until
    // we have full overflow checking implemented.
    //@ requires 0 <= p;
    //@ requires 0 <= q;
    //@ ensures false;
    public int non_trivial_arithmetic_overflow(int p, int q) {
        assert p + q >= 0;
        return p*q;
    }
}
