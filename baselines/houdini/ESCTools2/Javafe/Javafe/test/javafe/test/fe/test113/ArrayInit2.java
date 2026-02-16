/**
 ** Test anonymous array initializers
 **/


/*
 * Test basic parsing:
 */

class ArrayInit {
    // continued from ArrayInit.java

    // dims are not allowed with an initializer:
    Object obj = new int[12]{1, 2};     // parse error
}
