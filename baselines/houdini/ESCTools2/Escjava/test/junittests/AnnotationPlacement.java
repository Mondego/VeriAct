// Tests unconventional placement of JML annotations.

public class AnnotationPlacement
{

	public void m() //@ ensures true;
	{}

	public Object mm() //-@ non_null pure
	{ return null; }	

	public Object f /*-@ non_null */;
	public Object ff=null /*-@ non_null */;
	public int[] af={0,0,0} /*-@ non_null */;

	public Object p( Object o /*-@ non_null */, Object oo /*-@ non_null */ ) {

	    int i = 0 /*-@ uninitialized */;
	    Object ooo /*-@ non_null */;
	    return null;
	}
}
