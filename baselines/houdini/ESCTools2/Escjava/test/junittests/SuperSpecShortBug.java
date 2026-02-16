// Test case from Bug #430
//#FLAGS: -Q

public class SuperSpecShortBug
    extends CSuperSpecShortBug {
    // {

    //@ also
    /* @ public exceptional_behavior
      @    requires o != null;
      @    requires Object.class != o.getClass();
      @    signals_only IllegalArgumentException;
      @ also */
    /*@ public normal_behavior
      @    requires o != null;
      @    requires o.mmm() == b;
      @*/
    /*@pure*/ 
    public void m(P o) throws IllegalArgumentException 
    {
	if(o.mmm() != b)
		throw e;
	    //throw new IllegalArgumentException();
    }
}

class CSuperSpecShortBug {
    int b;

    /* @ public exceptional_behavior
      @    requires o != null;
      @    requires Object.class != o.getClass();
      @    signals_only IllegalArgumentException;
      @ also */
    /*@ public normal_behavior
      @    requires o != null;
      @    requires o.mmm() == b;
      @*/
    /*@pure*/ 
    public void m(P o) throws IllegalArgumentException 
    {
	if(o.mmm() != b)
		throw e;
	    //throw new IllegalArgumentException();
    }

    /*@ non_null */ public final static IllegalArgumentException e = new IllegalArgumentException();
}

class P {
    //@ pure
    int mmm() { return 0; }
}
