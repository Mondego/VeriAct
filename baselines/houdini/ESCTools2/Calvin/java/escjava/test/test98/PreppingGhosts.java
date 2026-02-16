/* Copyright Hewlett-Packard, 2002 */

/**
 ** Verify that ghost fields are prepped before they are used
 ** 
 **/

class PreppingGhosts1 {
    //@ invariant Ghosty1.x == null
}

class Ghosty1 {
    //@ ghost static public Object x
}



class PreppingGhosts2 {
    //@ invariant Ghosty2.y == null         // error
}

class Ghosty2 {
    //@ ghost static public Nonexistent y
}

