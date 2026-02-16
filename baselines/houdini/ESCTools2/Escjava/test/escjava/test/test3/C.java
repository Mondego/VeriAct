
public class C {
    void test0() {
        // skip
    }
    void test1() {
	int x=0;
	//@ assert x == 0;
    }
    void test2() {                 
	int x;
	//@ assert x == 0; // ERROR
    }
    void test3() {
	int x;
	//@ assume x == 0;
    }
    void test4() {
	int x=0;
	//@ assume x == 0;
    }
    void test5() {
	int x=0;
	//@ assert x == 1; // ERROR
    }
    void test6() {
	int x=0;
	//@ assume x == 1;
    }
    void test7() {
	int x=0;
	//@ assume (\forall int i; i>0);
	//@ assume (\forall C c; c == c);
    }
}
