/**
 ** Test that we use the correct root interface
 **/

package java.lang;

class Object {

    public int pub;
    /*package*/ int pac;
    protected int prot;
    private int pri;

    public int getPub() { return pub; }
    /*package*/ int getPac() { return pub; }
    protected int getProt() { return pub; }
    private int getPri() { return pub; }

    synchronized public void foo() {};
    public static void bar() {};
    public final void baz() {};
    public native void quux();
}

interface I {}
interface J extends I {}


class Test {

    void passable(Object x) {}

    void test(J i) {
	passable(i);

	i.getPub();
	i.getPac();          // error
	i.getProt();         // error
	i.getPri();          // error

	i.foo();
	i.bar();             // error
	i.baz();
	i.quux();

	int x1 = i.pub;      // error
	int x2 = i.pac;      // error
	int x3 = i.prot;     // error
	int x4 = i.pri;      // error
    }
}
