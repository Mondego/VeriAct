// Tests parsing and use of model classes and interfaces

public class ModelClass {

/*@
	public model class M {}
	private model class MP {}
	model public interface I {}
*/

	//@ invariant new M() != null;

	//@ model public void m(I i) {}

	//@ invariant new MP() != null;
}
