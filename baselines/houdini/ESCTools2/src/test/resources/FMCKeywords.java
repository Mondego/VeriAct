// Tests the field, method and constructor keywords


public class FMCKeywords {

	//@ ghost requires j; // ERROR
	//@ ghost field requires j; // OK

	//@ model requires j; // ERROR
	//@ public model field requires j; // OK

	//@ public model requires ensures() { int signals = 0; }  // ERROR
	//@ public model method requires ensures() { int signals = 0; } // OK

}

class requires {
	public requires() {} // OK
	//@ public model requires(Object o) {} // ERROR
	//@ public model constructor requires(Object o) {} // OK
	//@ public model constructor int() {} // ERROR
	//@ public model constructor requires[]() {} // ERROR
	//@ public model constructor sub.badname() {} // ERROR

	public R() {}
}
