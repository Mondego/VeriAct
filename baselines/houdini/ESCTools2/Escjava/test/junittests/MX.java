/* Checks for the error that the method modifier was applied to all of the 
   annotations in the body.
*/
public class MX {

	protected int i;

/*@
	requires i > 0;

	private model int mm() {
		assume i > 0;
		maintaining i > 0;
		while (i < 10) i++;
		assert true;
	}
*/
}
