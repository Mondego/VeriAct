public class Exc {

    public int i;

    //@ requires true;
    //@ ensures false;
    //@ signals (Exception e) true;
    //@ pure
    public  boolean  mm() throws Exception { 
        throw new Exception(); 
    }

    //@ requires mm();
    public void m() { }    

    //@ ensures mm();
    public void m2() { }    

    //@ requires o.i == 0;
    public void p(Exc o) {}
}
