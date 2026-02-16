public class ExceptionArgument {
	public final String n = "NO";

	//@ signals (IllegalArgumentException e) e.getMessage().equals(n);
	public void m() throws IllegalArgumentException {
		throw new IllegalArgumentException(n);
	}
}
