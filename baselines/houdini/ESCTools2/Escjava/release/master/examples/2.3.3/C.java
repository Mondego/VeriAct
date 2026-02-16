class C {

    int f;

    //@ ensures f == \old(f) + 1;    
    void incf() {
      this.f++;
    }

    void p() {
      incf();
      //@ assert false;
    }

}
