/* Copyright Hewlett-Packard, 2002 */

class D {

    static D someOtherObject;

    int f;

    //@ modifies someOtherObject.f; //instead of this.f
    //@ ensures f == \old(f) + 1;    
    void incf() {
      this.f++;
    }

    //@ requires this != someOtherObject;
    void p() {
      incf();
      //@ assert false;
    }

}
