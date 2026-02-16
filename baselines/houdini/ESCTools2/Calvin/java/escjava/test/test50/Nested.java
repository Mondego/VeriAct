/* Copyright Hewlett-Packard, 2002 */

/**
 ** This file tests parsing of nested escjava annotations
 **/

class Nested {

    /*
     * level 1:
     */

    //@ ghost public int[] t //@ non_null
    //@ ghost public /*@ non_null @*/ int[] x
    //@ ghost public /**<esc>non_null</esc>*/ int[] y

    /*@ ghost public int[] v //@ non_null */
    //@ ghost public int[] w /** <esc>non_null</esc> */
    //@ ghost public int[] a /** <esc></esc> */

    //*<esc> ghost public int[] b /*@ non_null @*/ </esc>

    /*@ invariant a[1] > a[0] ; //@ nowarn
      invariant a[2] > a[1]  //@ nowarn
    */

    //@ ghost public /*@ non_null */ /*@ readable_if a[0] > 0 */ Nested r 

    /*
     * level 2:
     */

    //@ ghost public int[] e //* non_null //@non_null
}
