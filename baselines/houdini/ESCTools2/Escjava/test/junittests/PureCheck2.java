// Checks the pure declarations on class and interface
public class PureCheck2 extends SuperPure implements IPure {
	//@ requires spm() + ipm() + npm() > 0;  // ERROR -needs also
	//@ pure
	public int ipm() { return 0; }

}

//@ pure
class SuperPure extends NonPure {
	//@ ensures \result == 1;
	public int spm(){ return 1; }
}

/*@ pure @*/ interface IPure {
	//@ ensures \result == -2;
	public int ipm();
}

class NonPure {
	public int npm() { return 2; }
}
