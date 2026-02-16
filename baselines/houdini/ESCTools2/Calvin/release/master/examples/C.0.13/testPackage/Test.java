/* Copyright Hewlett-Packard, 2002 */

package testPackage;

class Test {
    public static void main(String[] args) {
        String hello = "Hello", lo = "lo";
        String temp;

/* As explained in [JLS, 3.10.5], the following statements produce the output
   "true true true true false true". */

        System.out.print((hello == "Hello") + " ");
        System.out.print((Other.hello == hello) + " ");
        System.out.print((other.Other.hello == hello) + " ");
        System.out.print((hello == ("Hel"+"lo")) + " ");
        System.out.print((hello == ("Hel"+lo)) + " ");
        System.out.println(hello == ("Hel"+lo).intern());

	/* The following assertions are all correct, but give rise to
           warnings due to incompleteness in ESJ/Java's semantics
           for strings. */

	//@ assert hello == "Hello";           // correct but fails
        //@ assert Other.hello == hello;       // correct but fails
        //@ assert other.Other.hello == hello; // correct but fails
        //@ assert hello == ("Hel"+"lo");      // correct but fails
        //@ assert hello != ("Hel"+lo);        // correct but fails
        temp = ("Hel"+lo).intern();
        //@ assert hello == temp;              // correct but fails

    }

}

class Other { static String hello = "Hello"; }

