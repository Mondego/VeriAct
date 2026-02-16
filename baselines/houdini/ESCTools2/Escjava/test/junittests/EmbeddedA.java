// Tests some improperly formed annotations in javadoc comments.

public class EmbeddedA {

	/** Doc.
	<jml>
		ensures false;
	</esc>
	*/ // mismatched end tag
	public void m() ;

	/** Doc.
	<JML>
		ensures false;
	*/ // No end tag
	public void n() ;

	/** Doc.
	<JML>
		ensures false;
	<
	*/ // No end tag
	public void n1() ;

	/** Doc.
	<JML>
		ensures false;
	</JML
	*/ // No end tag
	public void n2() ;

	/** Doc.
	<ES
	<JML>
		ensures false; // Should read this one
	</JML>
	*/ // False start
	public void p() {}

	/** Doc.
	<pre> XXX
	<JML>
		ensures false; // Should read this one
	</JML>
	</pre>
	*/ // other tags
	public void p2() {}
}

