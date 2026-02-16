// Tests the informal predicate

public class Informal {

	public int i;

	//@ requires (* ok *);
	//@ diverges (* ok *);
	//@ ensures (* ok *);
	//@ ensures ! (* ok *);
	//@ modifies (* stuff *), i;
	//@ signals (Exception e) (* ok *);
	//@ when (* ok *);
	//@ duration (* ok *) ? 0:0;
	//@ working_space (* ok *)?0:0;
	//@ pre (* ok *);
	//@ post (* ok *);
	//@ assignable (* ok *) ;
	//@ also code_contract
	//@ accessible (* ok *);
	//@	measured_by (* ok *);
	//@	callable (* ok *);
	public void m() throws Exception {}

}
