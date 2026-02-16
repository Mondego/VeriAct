// Checks that the desugaring of pure works ok

public class PureCheck3 {
    public boolean b;

    public void m() {}

    //@ pure
    public void mp() {}

    //@ requires b;
    public void n() {}


    //@ requires b;
    //@ pure
    public void np() {}


    public void a() {
	b = false;
	n();    // WARNING
    }
    public void ap() {
	b = false;
	np();   // WARNING
    }

    //@ modifies \nothing;
    public PureCheck3() { }
}

class PureCheck3A extends PureCheck3 {

    public boolean b;

    //@ also
    //@ requires b;
    public void mp() {}

    //@ pure
    public void n() {}

    public void a() {
	b = false;
	mp();
    }
    public void ap() {
	b = false;
	n();    // WARNING
    }
}
