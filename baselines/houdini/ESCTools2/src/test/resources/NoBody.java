// Tests that escjava handles methods with no body.

public class NoBody {
	//@ modifies \nothing;
	public NoBody m();
	public void n() {
	    m().m();
	}
	public NoBody();
}
