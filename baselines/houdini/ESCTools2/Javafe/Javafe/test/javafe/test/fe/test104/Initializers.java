/**
 ** Test type checking of initializer bodies (static & instance)
 **/

class Initializers {

    int i;
    static int s;

    // Test static initializers:
    static {
	s = s;
	s = i;          // Error: this is a static context
    }

    // Test instance initializers [1.1]:
    {
	s = s;
	s = i;
    }

    // Make sure we check the use of modifiers on initializers:
    final { }           // Error: bad modifier
}
