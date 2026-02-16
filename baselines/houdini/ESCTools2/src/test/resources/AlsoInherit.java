// This test checks whether we get warnings because of misuse of 'also'

public class AlsoInherit extends Parent {

	//@ ensures true;
	public void m();  // ERROR - no also

	//@ ensures true;
	public void n();

	//@ also
	//@ ensures true;
	public void malso();

	//@ also		// ERROR - should not have an also
	//@ ensures true;
	public void nalso();

	//@ also_ensures true;  // OBSOLETE
	public void p();	// ERROR - should have an also

	//@ also_exsures (Exception) true; // OBSOLETE
	public void pp() throws Exception;	

}

class Parent {
	//@ ensures true;
	public void m();

	//@ also		// ERROR - no also expected
	//@ ensures true;
	public void malso();

	public void p();
}
