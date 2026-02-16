/**
 ** This file tests that we check that inner classes members may not
 ** be static, part II
 **/

// test case for nested interfaces as well, continued:
class OutsideClass4 { 
    interface Nested {
	static { }                     // error (no initializers in interfaces)
    }
}
