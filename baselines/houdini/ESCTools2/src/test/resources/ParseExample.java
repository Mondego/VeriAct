public class ParseExample {

	/*@
		requires true;
		ensures true;

	implies_that
		requires i>0;
		ensures true;
	for_example
		requires i == 1;
		ensures i == 1;
	also
	     public normal_example
		requires i == 5;
		ensures i == 6;
	also protected exceptional_example
		requires i<0;
		signals (Exception) false;
	also private example
		requires i==0;
		ensures i==0;
	*/
	public void m(int i);
}
	
