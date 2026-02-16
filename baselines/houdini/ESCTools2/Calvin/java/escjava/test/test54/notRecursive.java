/* Copyright Hewlett-Packard, 2002 */

class C {
    
    /*@ helper */ private void m() {
	q();
    }

    /*@ helper */ final void q() {
	r();
    }
    
    final void r() {
	q();
    }
}
