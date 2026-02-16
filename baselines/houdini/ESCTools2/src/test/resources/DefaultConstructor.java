// Tests that the absence of constructors in the jml file does not cause
// spurious error messages or additional constructors.
//@ refine "DefaultConstructor.jml";
public class DefaultConstructor {

	public DefaultConstructor(int i) {}
}
