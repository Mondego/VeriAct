// Tests that escjava can read stuff within annotations embedded in javadoc
// comments - for any of the four allowed tags.
// We know it can read it when there is an error.
public class Embedded {

	/** Doc.
	<esc>
		ensures false;
	</esc>
	*/
	public void m() {}

	/** Doc.
	<jml>
		ensures false;
	</jml>
	*/
	public void n() {}

	/** Doc.
	<ESC>
		ensures false;
	</ESC>
	*/
	public void q() {}

	/** Doc.
	<JML>
		ensures false;
	</JML>
	*/
	public void p() {}

// Tests whether multiple annotations within one javadoc comment work.


	/** Doc.
	<esc>
		requires true;
	</esc>

	More doc.

	<jml>
		ensures false;
	</jml>
	*/
	public void mm() {}
}
