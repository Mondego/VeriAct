class MiscFields {

    //@ invariant y!=0;
    static int y = 1;
}


/*
 * Test that we see references from direct instance fields when checking
 * constructors
 */


class MiscFieldsUser1 {

    //@ invariant x!=0;
    int x = MiscFields.y;


    MiscFieldsUser1() {}		// no error
}

class MiscFieldsUser2 {

    //@ invariant x==0;
    int x = MiscFields.y;


    MiscFieldsUser2() {}		// error
}


/*
 * Test that we see references from referenced field specs
 */

class MiscFieldsUser3 {

    /*@ readable_if MiscFields.y!=0; */    int x;

    void foo() {
	System.out.println(x);				// no error
    }
}

class MiscFieldsUser4 {

    /*@ readable_if MiscFields.y==0; */    int x;

    void foo() {
	System.out.println(x);				// error!
    }
}
