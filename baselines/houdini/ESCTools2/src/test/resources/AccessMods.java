public class AccessMods {

	//@ public invariant f > 0;
	//@ protected invariant f > 0;
	//@ private invariant f > 0;
	public int f = 1;

	//@ requires i > 0;
	//@ ensures \result > 0 ;

	public int f(int i) { return i; }
}
