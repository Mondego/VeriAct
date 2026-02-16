/* Copyright Hewlett-Packard, 2002 */


class C {
    int i;
    
    void m0(Integer l) {
	int q = l.intValue();
	m1(l);
    }

    void m1(Integer k) {
	m2(k);
    }

    /*@ helper */ private void m2(Integer j) {
	i = j.intValue();
    }

	
}
