// Tests model constructors

public class ModelConstructors {

	//@ model constructor badname(int i) {  } // ERROR
	//@ model constructor sub.badname(float i) {  } // ERROR

	//@ model public int m(Object xx) { }

	/*@ model ModelConstructors(int i,int j) { 
		o == null;  // OK
		ooo == null; // ERROR
	}
	*/

	//@ ghost Object o;

	// NOTE - A default constructor should be produced even if a model
	// constructor is defined.

	public void mm() {
		//@ set o = new ModelConstructors(0,0);
	}
	public void mm2() {
		//@ set o = new ModelConstructors();
	}

	Object po;

/*@	model public void p() {
		o == new ModelConstructors();
		o == new ModelConstructors(0,0);
		o == po;
		ooo == null; // ERROR
	}
*/
}

class ModelConstructorZ {
	public void mm4() {
		Object oo = new ModelConstructors();
		oo = new ModelConstructors(0,0); // FIXME - ERROR - model constructor
		oo = new ModelConstructors(0,0,0); // ERROR - no such constructor
	}
}

class ModelConstructorA {

	//@ ghost Object o;

	public ModelConstructorA() { o = null; } // ERROR

	public void m() { o = null; } // ERROR

}

class ModelConstructorB {

	public ModelConstructorB(int i) {}

	//@ model public ModelConstructorB(int i) {}  // ERROR
}
