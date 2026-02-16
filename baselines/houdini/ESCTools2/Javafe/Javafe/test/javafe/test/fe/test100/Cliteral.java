/**
 ** This class tests the parsing of class literals (a 1.1 feature)
 **
 ** It should parse ok.
 **/

class Cliteral {
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

    static void out(Class c) {
	System.out.println(c);
    }
}
