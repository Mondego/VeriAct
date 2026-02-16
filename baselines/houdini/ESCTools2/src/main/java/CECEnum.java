/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.prover;


import java.util.*;

import javafe.util.*;


/**
 ** This class is used privately by Simplify to enumerate the list of
 ** counter-example contexts found by Simplify to a given verification
 ** condition. <p>
 **
 ** If the enumeration ends with a SimplifyOutput of kind VALID, then
 ** Simplify verified the theorem. <p>
 **
 **
 ** This class will eventually be fairly lazy to allow error reporting
 ** as each error is found.<p>
 **
 **
 ** Warning: This class uses SubProcess.readSExp(), which does not
 ** implement the full grammer for SExps.  See that routine for the
 ** exact limitations.<p>
 **/

class CECEnum implements Enumeration {

    /***************************************************
     *                                                 *
     * Instance variables:			       *
     *                                                 *
     ***************************************************/

    /**
     ** The Simplify subprocess. <p>
     **/
    private final /*@ non_null @*/ SubProcess P;


    /**
     ** Is Simplify done processing our verification request? <p>
     **
     ** If so, all remaining results must be in <code>pending</code>.
     ** Otherwise, the remaining results are those in <code>pending</code>
     ** followed by the results we have yet to read from Simplify.<p>
     **/
    //@ spec_public
    private boolean simplifyDone = false;

    /**
     ** The next results we are to return.
     **/
    //@ invariant pending.elementType == \type(SimplifyOutput);
    //@ invariant !pending.containsNull;
    //@ spec_public
    private final /*@ non_null */ Vector pending = new Vector();


    /***************************************************
     *                                                 *
     * Creation:                  		       *
     *                                                 *
     ***************************************************/

    /**
     ** Create an Enumeration of the counter-example contexts for
     ** expression <code>exp</code> using Simplify process
     ** <code>simplify</code>. <p>
     **
     ** We always return an Enumeration of non-null SimplifyOutput's.
     ** If verifying <code>exp</code> succeeds, the resulting Enumeration
     ** will end with a SimplifyOutput of kind VALID. <p>
     **
     ** Precondition: <code>simplify</code> is not closed, and
     **               <code>exp</code> is properly formed according to
     **               Simplify's rules.<p> 
     **
     ** Note: This routine should be called *only* from class Simplify.<p>
     **/
    /*package*/ CECEnum(/*@ non_null @*/ SubProcess simplify,
			/*@ non_null @*/ String exp) {
	//@ set pending.elementType = \type(SimplifyOutput);
	//@ set pending.containsNull = false;

	P = simplify;
	P.resetInfo();

	if (Info.on) {
	  Info.out("[calling Simplify on '" + exp + "']");
	}
	P.send(exp);

    }

    /*package*/ CECEnum(/*@ non_null @*/ SubProcess simplify) {
	//@ set pending.elementType = \type(SimplifyOutput);
	//@ set pending.containsNull = false;

	P = simplify;
	P.resetInfo();

	if (Info.on) {
	    Info.out("[calling Simplify on VC stream]");
	}
    }

    /***************************************************
     *                                                 *
     * Implementing the Enumeration interface:	       *
     *                                                 *
     ***************************************************/

    //@ represents moreElements <- !(simplifyDone && pending.elementCount == 0);

    //@ invariant_redundantly pending.elementCount > 0 ==> moreElements;
    //@ invariant_redundantly moreElements ==> !simplifyDone || pending.elementCount > 0;


    /**
     ** The type of the elements we return.
     **/
    //@ invariant elementType==\type(SimplifyOutput);

    /**
     ** Do we ever return null as an element?
     **/
    //@ invariant !returnsNull;


    /**
     ** Returns true iff any more elements exist in this enumeration.
     **/
    //@ also modifies simplifyDone, pending.elementCount;
    //@      ensures \result ==> pending.elementCount>0;
    final public boolean hasMoreElements() {
	if (pending.size()==0 && !simplifyDone)
	    readFromSimplify();

	return pending.size()!=0;
    }

    /**
     ** Returns the next element of the enumeration. Calls to this
     ** method will enumerate successive elements.  Throws
     ** NoSuchElementException if no more elements are left.
     **/
    //@ also modifies simplifyDone;
    public /*@ non_null */ Object nextElement() /*throws NoSuchElementException*/ {
	if (!hasMoreElements())
	    throw new NoSuchElementException();

	Object result = pending.firstElement();
	pending.removeElementAt(0);

	if (Info.on) {
	  Info.out("  [Simplify returned: " + result.toString() + "]");
	}

	return result;
    }


    /***************************************************
     *                                                 *
     * Reading results from Simplify:		       *
     *                                                 *
     ***************************************************/

    /**
     ** Ensure that we are done using Simplify.  We promise not to
     ** ever use Simplify again after this routine returns.
     **/
    /*package*/ void finishUsingSimplify() {
	while (!simplifyDone)
	    readFromSimplify();
    }


    /**
     ** Attempt to read another output (for example, a counter-example)
     ** from Simplify, then append it to <code>pending</code>.
     **/
    //@ requires !simplifyDone;
    //@ modifies simplifyDone, pending.elementCount;
    //@ ensures simplifyDone || pending.elementCount > 0;
    private void readFromSimplify() {
	P.eatWhitespace();

	SimplifyOutput so;
	if (Character.isDigit((char)P.peekChar())) {
	  so = readSentinel();
	  simplifyDone = true;
	} else {
	  so = readResultMessage();
	}
	pending.addElement(so);
    }

    private /*@ non_null @*/ SimplifyOutputSentinel readSentinel() {
      int n = P.readNumber();
      P.checkString(": ");

      int kind;
      switch (P.peekChar()) {
	case 'V':
	  P.checkString("Valid.");
	  kind = SimplifyOutput.VALID;
	  P.eatWhitespace();
	  if (P.peekChar() == '(')
		P.checkString("(Added to background predicate).");
	  break;
	case 'I':
	  P.checkString("Invalid.");
	  kind = SimplifyOutput.INVALID;
	  break;
	case 'U':
	  P.checkString("Unknown.");
	  kind = SimplifyOutput.UNKNOWN;
	  break;
	default:
	  P.handleUnexpected("'Valid', 'Invalid', or 'Unknown'");
	  return null;  // dummy return
      }
      P.eatWhitespace();

      return new SimplifyOutputSentinel(kind, n);
    }

    //@ ensures \result != null;
    private /*@ non_null */ SimplifyResult readResultMessage() {
      String msg = P.readWord("\n");
      P.checkChar('\n');
      SimplifyResult result = null;
      int kind;
      if (msg.startsWith("Counterexample:")) {
	kind = SimplifyOutput.COUNTEREXAMPLE;
      } else if (msg.startsWith("Exceeded PROVER_KILL_TIME")) {
	kind = SimplifyOutput.EXCEEDED_PROVER_KILL_TIME;
      } else if (msg.startsWith("Exceeded PROVER_KILL_ITER")) {
	kind = SimplifyOutput.EXCEEDED_PROVER_KILL_ITER;
      } else if (msg.startsWith("Reached PROVER_CC_LIMIT")) {
	kind = SimplifyOutput.REACHED_CC_LIMIT;
      } else if (msg.startsWith("Exceeded PROVER_SUBGOAL_KILL_TIME")) {
	kind = SimplifyOutput.EXCEEDED_PROVER_SUBGOAL_KILL_TIME;
      } else if (msg.startsWith("Exceeded PROVER_SUBGOAL_KILL_ITER")) {
	kind = SimplifyOutput.EXCEEDED_PROVER_SUBGOAL_KILL_ITER;
      } else if (msg.startsWith("Warning: triggerless quantifier body")) {
	SExp e0 = P.readSExp();
	P.eatWhitespace();
	P.checkString("with ");
	int n = P.readNumber();
	String restOfLine = P.readWord("\n");
	SExp e1 = P.readSExp();
	result = new TriggerlessQuantWarning(null, null, e0, n, e1);
	kind = result.getKind();
      } else {
	P.handleUnexpected("result message");
	return null;  // dummy return
      }

      if (result == null) {
	result = new SimplifyResult(kind, null, null);
      }

      P.eatWhitespace();

      // Read in the labels, if any
      if (P.peekChar() == 'l') {
	P.checkString("labels:");
	result.labels = P.readSList();
	P.eatWhitespace();
      }

      // Read in the set of counterexample contexts, if present
      if (P.peekChar() == 'c') {
	P.checkString("context:\n");
	result.context = P.readSList();
	P.eatWhitespace();
      }

      return result;
    }
}
