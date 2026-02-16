/* Copyright Hewlett-Packard, 2002 */

package escjava.prover;

import escjava.translate.UniqName;
import javafe.util.Location;


/** An object of this class represent a "result" produced by Simplify.
 ** <p>
 ** 
 ** @see Simplify
 ** @see CECEnum
 ** @see SExp
 **/

public class TriggerlessQuantWarning extends SimplifyResult {
  public /*@ non_null */ SExp e0;
  int n;
  //@ invariant 0 <= n;
  public /*@ non_null */ SExp e1;

  //@ requires 0 <= n;
  TriggerlessQuantWarning(SList labels, SList context,
			  /*@ non_null */ SExp e0, int n,
			  /*@ non_null */ SExp e1) {
    super(WARNING_TRIGGERLESS_QUANT, labels, context);
    this.e0 = e0;
    this.n = n;
    this.e1 = e1;
  }

  public String toString() {
    return super.toString() + " e0=" + e0 + " n=" + n + " e1=" + e1;
  }

  /** Attempts to glean a location from the name of the dummy variable
    * appearing in <code>e1</code>.  If none can be retrieved, the null
    * location is returned.
    **/
  public int getLocation() {
    try {
      SList quant = e1.getList();
      if (quant.length() < 2 || ! quant.at(0).toString().equals("FORALL")) {
	return Location.NULL;
      }
      SList dummies = quant.at(1).getList();
      String dummy = dummies.at(0).getAtom().toString();

      int k = dummy.indexOf(':');
      if (k == -1) {
	return Location.NULL;
      }
      return UniqName.suffixToLoc(dummy.substring(k+1), true);

    } catch (SExpTypeError sete) {
      return Location.NULL;
    }
  }
}
