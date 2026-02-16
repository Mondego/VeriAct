/* Copyright Hewlett-Packard, 2002 */

class C {

    static void main(String[] args) {
	//@ assert 1.0 == 1.0;                   // correct but fails
	//@ assert 1.0 != 2.0;                   // correct but fails
	//@ assert 1.0 + 1.0 == 2.0;             // correct but fails
	//@ assert "Hello world" != null;        // correct and passes
        String s = "Hello world";              
        char c = s.charAt(3);
        c = "Hello world".charAt(3);
	//@ assert c == 'l';                     // correct but fails
        System.out.println(c == 'l');
	//@ assert 2 + 2 == 4;                   // correct and passes
	//@ assert 2000000 < 4000000;            // correct and passes
	//@ assert 2000000 + 2000000 == 4000000; // correct but fails
    }

}
