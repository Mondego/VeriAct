// Tests the forall and old pragmas

public class ForallOldError {


	//@ forall Object o = null ; // ERROR
	//@ old Object x;   // ERROR
	public void z() {}

	//@ forall int x,y=4,z; // ERROR
	//@ old int a=1,b,c=3;  // ERROR
	//@ requires x+z == a+c;
	//@ requires b==y;  // ERROR
	public void zz() {}

}
