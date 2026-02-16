// Tests some errors unique to constructors


public class ConstructorErrors {

	/*@
		requires k==0;
		requires j == 0; // Error here because j is instance variable
		modifies j,k;
		ensures j == i;
		ensures k == 1;
	subclassing_contract 
		measured_by k;
		callable m();
	*/


	public Constructor(int i) { k++; j=i; }

	public int j;
	static public int k = 0;

	public void m() {}
}
