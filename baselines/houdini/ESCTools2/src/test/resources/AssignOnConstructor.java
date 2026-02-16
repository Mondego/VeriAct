
public class AssignOnConstructor {

	//@ model public int ostate;
	//@ ghost int x;

	//@ assignable x,ostate;
	public void m();

	//@ assignable x,ostate;
	public AssignOnConstructor();
}
