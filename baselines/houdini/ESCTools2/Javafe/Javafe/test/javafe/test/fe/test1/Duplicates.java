package p;

/*
 * This test checks to make sure a Tool handles a class defined twice in
 * the same file (as opposed to in separate files) properly.  It should
 * produce an error.
 */

class Duplicates {}

class Duplicates {}	// This is a duplicate definition and hence in error!
