public class ParseModelMethod {

/*@
    public model int m() { return 1; }

	model protected void q(int j) throws Exception {}

	ghost int i ;

	model private float h;
*/

	//@ requires true;
	void mm();

	//@ requires i == 0;
	void p();
}
