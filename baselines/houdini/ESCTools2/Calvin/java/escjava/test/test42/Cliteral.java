/* Copyright Hewlett-Packard, 2002 */

/**
 ** This class tests the translation of class literals.  See front end
 ** test fe/test100 for tests of parsing and typechecking of class
 ** literals (a 1.1 feature)
 **/

class Cliteral {
    void specTest() {
	//@ assert \TYPE.class == \TYPE.class
	//@ assert void[].class == void[].class
	//@ assert Cliteral.class == Cliteral.class
	//@ assert Cliteral[].class == Cliteral[].class
	//@ assert Cliteral[][].class == Cliteral[][].class
    }

    void typingTest() {
	//@ assert void.class!=null
	//@ assert \typeof(int.class) == \type(Class)
    }

    public static void main(String[] args) {

	// void . class:
	out(void.class);

	// <java primitive type> . class:
	out(int.class);
	out(short.class);
	out(byte.class);
	out(boolean.class);
	out(char.class);
	out(float.class);
	out(double.class);
	out(long.class);

	// <java primitive type> []* . class:
	out(int.class);
	out(int[].class);
	out(int[][].class);
	out(int[][][].class);
	out(void[].class);

	// <typename> . class:
	out(String.class);
	out(java.lang.Math.class);
	out(Cliteral.class);

	// <typename> []* . class:
	out(String[].class);
	out(java.lang.Math[].class);
	out(String[][].class);
	out(java.lang.Math[][].class);
    }

    void stillWorking() {
	Object[] objects = null;
	Object[] moreObjects = null;
    }

    static void out(/*@ non_null */ Class c) {
	System.out.println(c);
    }
}
