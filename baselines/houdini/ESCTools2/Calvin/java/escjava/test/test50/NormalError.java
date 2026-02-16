/* Copyright Hewlett-Packard, 2002 */

/**
 ** This file tests parsing of erroneous, normal escjava annotation comments
 **/

class NormalError {

    void wizard() {
	/*@(fobar*/      // error
    }
}
