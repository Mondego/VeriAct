// Tests parsing depends clauses

public class ParseDepends {

	//@ public model boolean b;

	boolean bb,bbb;

	//@ depends b <- bb,bbb;

	void m();
}
