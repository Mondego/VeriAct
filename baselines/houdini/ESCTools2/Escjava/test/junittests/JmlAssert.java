public class JmlAssert
{
    public void trivial() {
        //@ assert true;
    }

    public void constant_boolean_expression() {
        //@ assert true == true;
    }

    //@ requires p >= 0;
    public void assert_matches_precondition(int p) {
        //@ assert p >= 0;
    }

    //@ requires p >= 0;
    public void assert_weaker_than_precondition(int p) {
        //@ assert p >= -1;
    }
    
    //@ requires 0 == p;
    //@ requires 10 == q;
    public void constant_arithmetic(int p, int q) {
        //@ assert p + q == 10;
    }

    // Note that without the upper bounds on p and q this test should
    // fail due to possible overflow.
    //@ requires 0 <= p && p <= 1000;
    //@ requires 0 <= q && q <= 1000;
    public void non_trivial_arithmetic(int p, int q) {
        //@ assert p + q >= 0;
    }
}

