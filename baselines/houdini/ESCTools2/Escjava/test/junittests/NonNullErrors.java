// $Id$
// This file covers both non_null and nullable keywords 
public class NonNullErrors {

	//@ non_null
	public void mm();

	//@ non_null
	public int mmm();

	public void mmmm(/*@ non_null */ int i);

	//@ nullable
	public void mm2();

	//@ nullable
	public int mmm2();

	public void mmmm2(/*@ nullable */ int i);

}
