/* Copyright Hewlett-Packard, 2002 */

/**
 ** This file tests parsing of nested escjava annotations
 **/

class NestedError {

    /*
     * level 2:
     */

    //@ ghost public int[] c //@ non_null //@non_null // error
    //@ ghost public int[] d //@ non_null //*non_null // error
    //@ ghost public int[] e //* non_null //@non_null

    NestedError() {
      int[] a = new int[10];
      //@ set c = a;
      //@ set d = a;
    }
}
