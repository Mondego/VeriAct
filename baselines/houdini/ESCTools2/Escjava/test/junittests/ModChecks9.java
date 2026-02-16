public class ModChecks9 {
    public boolean P,Q;
    public int i,j,k;

    /*@ requires P;
	modifies i,j;
    also
	requires Q;
        modifies i,k;
    */
    public void m();

    //@ requires P && !Q;
    //@ modifies i;
    public void mi() { m(); }  // Not allowed to modify j

    //@ requires P && !Q;
    //@ modifies j;
    public void mj() { m(); } // Not allowed to modify i

    //@ requires P && !Q;
    //@ modifies i,j;
    public void mij() { m(); } // OK

    //@ requires P && !Q;
    //@ modifies k;
    public void mk() { m(); } // Not allowed to modify i,j

    //@ requires P && !Q;
    //@ modifies i,k;
    public void mik() { m(); } // Not allowed to modify j

    //@ requires P && !Q;
    //@ modifies i,j,k;
    public void mijk() { m(); } // OK

    //@ requires P && Q;
    //@ modifies i;
    public void pi() { m(); }  // OK - this is the tricky one

    //@ requires P && Q;
    //@ modifies j;
    public void pj() { m(); }  // Not allowed to modify i - twice,k

    //@ requires P && Q;
    //@ modifies k;
    public void pk() { m(); } // Not allowed to modify i - twice,j

    //@ requires P && Q;
    //@ modifies i,j;
    public void pij() { m(); } // OK - k can't be modified by m

    //@ requires P && Q;
    //@ modifies i,k;
    public void pik() { m(); } // OK - j can't be modified by m

    //@ requires P && Q;
    //@ modifies i,j,k;
    public void pijk() { m(); } // OK

}
