/**
 ** This file tests a misc. things that we weren't handling correctly;
 ** these bugs were discovered by running the front end on JDK 1.2.
 **/


/*
 * A bunch of expression typing checks...
 */

class A {
    // Make sure handle typing of ? : properly...
    byte m2(boolean b, byte x) { return b ? 0 : x; }
    byte m4(boolean b, byte x) { return b ? x : 0; }

    short s;
    short s2 = s++;       // check type is short, not int!
    short s3 = s--;       // ditto

    // The next test verifies that we do constant computation of casts
    // properly:
    private static final byte SI = (byte)0x80;   // This is -128, not 128!!
    private static final byte STOP = (byte) 0;
    private static final byte SI_STOP = (byte)SI + STOP;  // so this is ok...
}


/*
 * Make sure when checking throw clause compatibility, we take
 * checkedness of exceptions into account.
 */

class Throws {

    static public abstract class PutField {
	/**
	 * Put the value of the named boolean field into the persistent field.
	 */
	abstract public void put(String name, boolean value);
    }

    static final class PutFieldImpl extends PutField {
	 /**
	  * Put the value of the named boolean field into the persistent field.
	  */
	 public void put(String name, boolean value)
	     throws IllegalArgumentException
	 {
	     return;
	 }

    }
}
