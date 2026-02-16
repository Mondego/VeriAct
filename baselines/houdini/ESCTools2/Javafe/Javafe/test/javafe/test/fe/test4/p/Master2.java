package p;

/*
 * This test checks to make sure a Tool verifies the package a
 * type indirectly loaded from a source file claims to be in.
 */

// This class just serves to force indirectly loading the type Referenced2:
class Master extends Referenced2 {}
