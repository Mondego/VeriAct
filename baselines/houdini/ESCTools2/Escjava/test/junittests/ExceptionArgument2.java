public class ExceptionArgument2 {

	//@ signals (IllegalArgumentException e) e.getMessage().equals("NO");
	public void m() throws IllegalArgumentException {
		throw new IllegalArgumentException("NO");
	}
}
