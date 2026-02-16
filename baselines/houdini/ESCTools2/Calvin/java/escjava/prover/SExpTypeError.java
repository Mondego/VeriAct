/* Copyright Hewlett-Packard, 2002 */

package escjava.prover;


/**
 ** This checked exception is used to signal a dynamic "type error" in
 ** the use of S-expressions. <p>
 **
 ** E.g., it is thrown if a non-integer is treated as an integer, or an
 ** attempt is made to reference a non-existent element of a list.<p>
 **/

final public class SExpTypeError extends Exception {
}
