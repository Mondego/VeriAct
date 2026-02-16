/* Copyright Hewlett-Packard, 2002 */

class A {
    //@ ghost public int -> int -> int a;
    //@ ghost public static int -> boolean b;
    //@ ghost public static int -> char -> int c;
    //@ ghost public Object->A d;
    //@ ghost public static int -> boolean e;
    //@ ghost public String->A f;
    //@ ghost public Object[]->int g;
    //@ ghost public String[]->int h;

    //@ modifies c[2][*]
    void Test1() {
	char x;

	//@ set c[2][x] = 4;
    }

    void Test2() {
	//@ set a[2][3] = 4; 
	//@ set a[2][3] = 5; 

	//@ assert a[2][3] == 5
    }

    void Test3() {
	Object foo = new Object();

	//@ set d[foo] = this
	//@ set d[this] = this;
	//@ assert d[foo] == this
    }

    /* tests that maps are not aliased */
    void Test4() {
	int i;
	//@ set b = e;
	//@ set b[i] = true
	//@ set e[i] = false
	//@ assert b[i]
    }

    void Test5() {
	int i;
	//@ set b = e;
	//@ assert b[i] == e[i]
    }

    void Test6() {
	//@ set a[1][2] = 0;
	Test1();
	//@ assert a[1][2] == 0
    }
    
    //@ modifies a[2][*]
    void Test7() {
	//@ set a[2][3] = 1
    }

    void Test8() {
	//@ set a[2][3] = a[2]
	//@ set b = e
	//@ set b = a
	//@ set d = f
	//@ set f = d
	//@ set g = h
	//@ set h = g
	//@ assert b == e
    }
}

