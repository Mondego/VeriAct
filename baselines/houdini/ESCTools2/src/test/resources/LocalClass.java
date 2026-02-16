// This tests that the use of a type name inside a local block-level 
// declaration of that type resolves ok.  It was a bug.
//#FLAGS: -nowarn Modifies
public abstract class LocalClass 
{
    public LocalClass() ;

	//@ requires o != null;
	//@ ensures \result == 0;
    public int cc(Object o) {
	LocalClass oo = (LocalClass)o;    // WARNING
	return 0;
    }

    protected final int dispatcherWrapMethods()
    {
	class MethodRecord implements Comparable {
	    public MethodRecord() ;

	    public int compareTo(Object o ) {
		MethodRecord other = (MethodRecord) o;
		return 0;
	    }
	};
	return 0;
    }

    public int x() {
	class A {
		public A() ;

		//@ requires i > 0;
		//@ ensures \result < 0;
		public int m(int i) { return -i; }
	}

	A a = new A();
	int k = a.m(-1);  // Warning - prrecondition
	return 0;
    }
    public int xx() {
	class A {
		public A() ;

		//@ requires i > 0;
		//@ ensures \result < 0;
		public int m(int i) { return -i; }
	}

	A a = new A();
	int k = a.m(1);
	//@ assert k < 0;  // OK
	//@ assert k > 0;  // Warning
	return 0;
    }

    static public int istatic = 0;
    static public int jstatic = 0;

    //@ requires istatic > 0;
    //@ requires jstatic > 0;
    public void xxx() {

	class BB  {
	    //@ invariant jstatic > 0;

	    //@ modifies istatic,jstatic;
	    public void b() {}
	};

	int i = 0;
	//@ assert istatic > 0;
	//@ assert jstatic > 0;
	(new BB()).b();
	//@ assert jstatic > 0;  // OK
	//@ assert istatic > 0;  // ERROR
	
    }	
}

