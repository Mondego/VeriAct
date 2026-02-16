// Tests the parsing of multiple ids in a single ghost or model field
// declaration
public class MultipleDecl {

	//@ model public String a,b;
	//@ ghost public int i = 4, j = 6;

	//@ invariant a == null || a != null;
	//@ invariant b == null || b != null;
	//@ invariant i + 2 == j;

	public void m();
}
