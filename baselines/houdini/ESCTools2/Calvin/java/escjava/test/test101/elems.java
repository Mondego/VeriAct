/* Copyright Hewlett-Packard, 2002 */

class C {

    int[] p;

    void foo() {
	p = new int[10];

	for(int i=0; i<10; i++) {
	    p[i]++;
	}

    }
}
