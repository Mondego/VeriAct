package p;

/*
 * This test checks to make sure a Tool verifies that an
 * indirectly-loaded source file contains the type that it is named
 * after.
 */

// This class just serves to force indirectly loading the type Referenced:
class Master extends Referenced {}
