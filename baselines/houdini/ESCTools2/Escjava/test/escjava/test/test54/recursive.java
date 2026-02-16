class C {
    
    /*@ helper */ private void m() {
	q();
    }

    private void n() {
        m();
    }

    /*@ helper */ final void q() {
	r();
    }
    
    /*@ helper */ final void r() {
	q();
    }
}
