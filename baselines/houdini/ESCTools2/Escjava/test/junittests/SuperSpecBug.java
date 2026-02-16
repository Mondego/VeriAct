// Test case from Bug #430
//#FLAGS: -Q

public class SuperSpecBug
    extends CSuperSpecBug {
    // {

    //@ also
    /*@ public exceptional_behavior
      @    requires o != null;
      @    requires Object.class != o.getClass();
      @    signals_only IllegalArgumentException;
      @ also public normal_behavior
      @    requires o != null;
      @    requires Object.class == o.getClass();
      @*/
    /*@pure*/ 
    public void m(Object o) throws IllegalArgumentException 
    {
	if(Object.class != o.getClass())
	    throw new IllegalArgumentException();
    }
}

class CSuperSpecBug {
    /*@ public exceptional_behavior
      @    requires o != null;
      @    requires Object.class != o.getClass();
      @    signals_only IllegalArgumentException;
      @ also public normal_behavior
      @    requires o != null;
      @    requires Object.class == o.getClass();
      @*/
    /*@pure*/ 
    public void m(Object o) throws IllegalArgumentException 
    {
	if(Object.class != o.getClass())
	    throw new IllegalArgumentException();
    }
}
