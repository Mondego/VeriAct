/* Copyright Hewlett-Packard, 2002 */

class CheckTransitivity {

    boolean a;
    int b;
    int c;
    int[] d;

    // REFLEXIVE+TRANSITIVE & {Stutter; OTI} is Transitive
    /* oti \old(a) ==> (b == \old(b) && a) */

    // REFLEXIVE BUT NOT TRANSITIVE
    /* oti (\old(a) == a) ==> (b == \old(b)) */

    // REFLEXIVE+TRANSITIVE but {Stutter; OTI} is not
    /* oti (c <= b) ==> (\old(b) >= b && \old(c) <= c) */
 
   /* oti (d[4] <= b) ==> (\old(b) >= b && \old(d[4]) <= d[4]) */

    // Not REFLEXIVE but TRANSITIVE, but 
    // {Stutter; OTI} is not Transitive
    /*@ oti (c < b) ==> (\old(b) > b && \old(c) < c) */

    /*@ modifies a, b */
    /*@ performs (a) {a == !\old(a)} */
    /* performs (b) {b == \old(b+1)} */
    void method1() {

    }

    /*@ modifies a, b */
    void method2() {
	method1();
    }

    /*@ modifies b, c, d[4] */
    /* performs (b) {true} */
    /* performs (d[4]) {\old(d[4]) == d[4]+1} */
    void method3() {
    }

    /*@ modifies b, c */
    void method4() {
	method3();
    }

}





