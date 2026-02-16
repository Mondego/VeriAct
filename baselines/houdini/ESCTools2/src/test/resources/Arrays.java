// Tests new array expressions within annotations.
//#FLAGS: -classpath . -quiet
public class Arrays {

	//@ ensures (new int[5]).length == 5;
	//@ ensures (new double[6][7]).length == 7;
	//@ ensures (new double[6][7])[0].length == 6;
	public void m() {}


	//@ ensures (new Arrays[2])[0] == null;
	//@ ensures (new int[2])[0] == 0;
	//@ ensures (new double[2])[0] == 0.0;
	//@ ensures (new int[2][3])[0][0] == 0;
	//@ ensures (new int[2][3])[0] != null;
	public void n() {}

	//@ ensures \typeof(new Arrays[2]) == \type(Arrays[]);
	//@ ensures \typeof(new int[2]) == \type(int[]);
	//@ ensures \typeof(new int[2][3]) == \type(int[][]);
	public void p() {}

	//@ ensures (new int[1]) != (new int[1]);
	public void q() {}

	int[][] ai;
	//@ requires (\forall int i; 0<=i && i <10 ==> ai[i] == new int[5]);
	//@ ensures ai[0] != ai[1];  // FIXME - not sure why this is established
	public void qq() {}
}
