/* Copyright Hewlett-Packard, 2002 */


class Test {
    //@ ghost /*@ elems_guarded_by[i] udder[i] == \tid */ public static int -> Thread udder

    //@ ghost /*@ elems_guarded_by[i] udder[i] == \tid */ public static int -> int cow

    //@ ensures \result == cow[i]
    static int geti(int i) { }

    static final int max = 3;
 
    /*@ modifies cow[\result];
        performs (cow[\result]) {
	   \result == -1 || 
	   (0 <= \result && \result < max
	   && \old(cow[\result]) == 0 && cow[\result] == 1) 
        }
    */
    static int bovine() {
	int i;
	for (i = 0; i < 3; i++) {
	    {{
		//@assume udder[i] == null 
		//@set udder[i] = \tid 
	    }}

	    if (geti(i) == 0) {
		//@ set cow[i] = 1
		//@ set \witness = 1
		//@ set udder[i] = null
		return i;
	    }

	    //@ set udder[i] = null
	}

	//@ set \witness = 1
	return -1;
    }

    /*@ 
      modifies cow[\result];
      performs (cow[\result]) {
      \result == 1 && \old(cow[\result]) == 0 && cow[\result] == 1
      }
    */
    static int bovine2() {
	int i = 0;

	{{
	    //@assume udder[i] == null 
	    //@set udder[i] = \tid 
	}}

	//@ assume cow[i] != 0

	//@ set udder[i] = null

	i++;

	{{
	    //@ assume udder[i] == null 
	    //@ set udder[i] = \tid 
	}}

	//@ assume cow[i] == 0

	//@ set cow[i] = 1

	//@ set udder[i] = null

	//@ assert false

	//@ set \witness = 1
	return i;
    }

    /*@ 
      modifies cow[\result];
      performs (cow[\result]) {
      cow[\result] == 1
      }
    */
    static int bovine3() {
	int i = 1;

	//@ set cow[i] = 1
	//@ set \witness = 1
	return i;
    }

    /*@ modifies cow[\result];
        performs (cow[\result]) {
	   \result == -1 || 
	   (0 <= \result && \result < max
	   && \old(cow[\result]) == 0 && cow[\result] == 1) 
        }
    */
    static int bovine4() {
	int i = 0;

	{{
	    //@assume udder[i] == null 
	    //@set udder[i] = \tid 
	}}

	if (geti(i) == 0) {
	    /*
	    //@ set cow[i] = 1
	    //@ set \witness = 1
	    */
	    //@ set udder[i] = null
	    return i;
	}

	//@ assert false
	/*
	//@ set udder[i] = null
	//@ set \witness = 1
	*/
	return -1;
    }

    static int bovine6() {
	int i;
	for (i = 0; i < max; i++) {
	    {{
		//@assume udder[i] == null 
		//@set udder[i] = \tid 
	    }}

	    if (geti(i) == 0) {

		//@ set udder[i] = null
		return i;
	    }

	    //@ set udder[i] = null
	}

	return -1;
    }

    static void  bovine7() {
	{{

	}}
    }
}
