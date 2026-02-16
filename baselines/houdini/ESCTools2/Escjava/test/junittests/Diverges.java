// Tests that a pure method may not have diverges clauses

public class Diverges {
	public boolean p;

    //@ diverges p;
    public void m();

    //@ requires p;  // A lightweight spec must have a diverges false
    //@ pure         // CAUTION
    public void m1();

    //@ requires p;
    //@ also
    //@ requires !p;
    public void m2light();

    //@ requires p;
    //@ also
    //@ requires !p;
    //@ diverges p;
    public void m1light();

    //@ diverges p;      // ERROR
    //@ pure
    public void m2();

    //@ public normal_behavior
    //@ requires p;
    //@ also public normal_behavior
    //@ diverges !p;     // ERROR
    //@ pure
    public void m3();

    //@ public normal_behavior
    //@ diverges false;
    //@ pure
    public void mm();

    //@ diverges false;
    //@ pure
    public void mmm();

}
