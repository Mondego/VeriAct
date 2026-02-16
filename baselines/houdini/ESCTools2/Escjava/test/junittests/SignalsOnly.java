public class SignalsOnly {

	static class A extends Exception {}
	static class B extends Exception {}


	//@ signals_only A,B;
	public void m(int i) throws A,B {
	  if (i == 0) throw new A();
	  if (i == 1) throw new B();
	}

	//@ signals_only A;    // WARNING (line 16)
	public void m1(int i) throws A,B {
	  if (i == 0) throw new A();
	  if (i == 1) throw new B();
	}

	//@ signals_only B;    // WARNING (line 21)
	public void m2(int i) throws A,B {
	  if (i == 0) throw new A();
	  if (i == 1) throw new B();
	}

        //@ requires i == 0;
	//@ signals_only A;  
        //@ also
        //@ requires i == 1;
	//@ signals_only B;  
	public void mp(int i) throws A,B {
	  if (i == 0) throw new A();
	  if (i == 1) throw new B();
	}

	//@ signals_only A,NullPointerException;
	public void mm(int i) throws A {
	  if (i == 0) throw new A();
	  if (i == 1) throw new NullPointerException();
	}
		

}
