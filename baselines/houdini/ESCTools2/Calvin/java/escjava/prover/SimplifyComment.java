/* Copyright Hewlett-Packard, 2002 */

package escjava.prover;

/** An object of this class represent a progress comment produced by Simplify.
 ** <p>
 ** 
 ** @see Simplify
 ** @see CECEnum
 ** @see SExp
 **/

public class SimplifyComment extends SimplifyOutput {
  final String msg;

  public String getMsg() {
    return msg;
  }

  SimplifyComment(String msg) {
    super(COMMENT);
    this.msg = msg;
  }
}
