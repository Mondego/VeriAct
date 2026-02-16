/**
 ** Test anonymous array initializers
 **/


/*
 * Test basic parsing:
 */

class ArrayInit {
    String[] o = { "hello", "there" };

    String[] s = new String[] { "good", "bye" };

    String[] e = new String[] {};

    String[][] d = new String[][] { { "a", "b" }, { "c" , "d" } };

    // continued in ArrayInit2.java...
}


/*
 * Test typechecking:
 */

class ArrayInit2 {
    int[] x = new int[] { 3, "hello" };    // error: type mismatch

    // test subtyping:
    Object[] o = new Object[] { "hello" };

    // 2-dim:
    String[][] d = new String[][] { { "a", "b" }, { "c" } };

    int[] s = new int[]{ 1, 2 };
    int[][] t = new int[][]{ s, { 3 }, {} };

    // wrong # of dims:
    Object too = new int[][]{ 1 };         // error
    Object few = new int[]{ { 2, 3 } };    // error
}
