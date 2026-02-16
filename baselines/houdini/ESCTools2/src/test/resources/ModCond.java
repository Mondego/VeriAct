public class ModCond {

    ModCond();

    public int j,k;

    /*@
        requires i > 0;
        modifies j;
    also
        requires i < 0;
        modifies k;
    */
    public void m(int i);

    //@ requires j ==0 && k == 0;
    public void m1(int i) {
        //@ assume i == 1;
        m(i);
        //@ assert j == 0;  // WARNING
    }

    //@ requires j ==0 && k == 0;
    public void m2(int i) {
        //@ assume i == 1;
        m(i);
        //@ assert k == 0;  // OK
    }

    //@ requires j ==0 && k == 0;
    public void m3(int i) {
        //@ assume i == -1;
        m(i);
        //@ assert j == 0; // OK
    }

    //@ requires j ==0 && k == 0;
    public void m4(int i) {
        //@ assume i == -1;
        m(i);
        //@ assert k == 0; // WARNING
    }


}
