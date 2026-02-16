public class Modifies {

	public int i;
	/*@
		pre i > 0;
		modifies i;
		modifiable i;
		assignable i;
		post i> 0;
		signals (Exception) false;
	*/
	public void m() throws Exception {}
}
