// Tests a constructor spec


public class Constructor {

	/*@
		requires k==0;
		modifies j,k,this.*;
		ensures j == i;
		ensures k == 1;
		callable m();
	*/


	public Constructor(int i) { k++; j=i; }

	public int j;
	static public int k = 0;

	public void m() {}
}
