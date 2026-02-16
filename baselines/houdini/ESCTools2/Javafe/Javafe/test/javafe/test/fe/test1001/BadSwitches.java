// Test type checking of switch statements

class BadSwitches {
    
    // Checking the type of the switch expression:
    void foo(byte b, char c, short s, int i, long l, Object o) {

	switch (b) {}
	switch (c) {}
	switch (s) {}
	switch (i) {}
	switch (l) {}    // error!

	switch (true) {}  // error!
	switch (1.0) {}   // error!
	switch ("hello") {}   // error!
	switch (o) {}    // error!
    }

    // misc checks:
    void foo() {
	switch (0) {
	case "hello":   // error
	case 14L:
	case true:      // error
	}

    }
}
