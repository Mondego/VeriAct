//#FLAGS: -parsePlus -classpath .
public class ParsePlus {

	//+@ ensures false;
	void m() {}

	/*+@
		ensures false;
	@+*/
	void n() {}

	/*+@
		ensures 4 == 2
	@+ 1;
	@*/
	void p() {}

	public /*@ pure @*/ void q() {}

	public /*+@ pure @+*/ void r() {}

	/*+@ ensures
	   @ false;     @+*/
	public void s() {}
}
