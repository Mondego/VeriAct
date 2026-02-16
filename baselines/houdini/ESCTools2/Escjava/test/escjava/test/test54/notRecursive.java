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
