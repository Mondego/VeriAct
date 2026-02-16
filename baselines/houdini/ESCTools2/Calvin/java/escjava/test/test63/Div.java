/* Copyright Hewlett-Packard, 2002 */

class Div {

    //@ requires lo <= hi;
    void m(int lo, int hi) {
	int mid = (lo + hi)/2;
	//@ assert lo <= mid;
	//@ assert mid <= hi;
    }
}
