// This checks that an assertion is connected with a model method
// (or at least that it does not get associated with the next
// non-model method).
public class ModelMethods2 {

	//@ ensures false;
	//@ model int m();

	//@ ensures 2 + 2 == 4;
	public void n() {}
}
