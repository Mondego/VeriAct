/* Copyright Hewlett-Packard, 2002 */

package escjava.prover;


import java.io.*;
import java.util.Enumeration;

import javafe.util.*;


/**
 ** The interface to the Simplify theorem prover program using
 ** Strings to send and S-expressions to receive. <p>
 **
 ** Note that a Java application will not exit normally (i.e., when its
 ** <code>main</code> method returns) until all subprocesses have
 ** finished.  In particular, this implies that all
 ** <code>Simplify</code> instances must be closed before this can
 ** occur.  Alternatively, a Java application can just call
 ** <code>java.lang.System.exit</code> to force an exit to occur without
 ** waiting for subprocesses.<p>
 **
 **
 ** Warning: This class uses SubProcess.readSExp(), which does not
 ** implement the full grammer for SExps.  See that routine for the
 ** exact limitations.<p>
 **
 ** @see SExp
 ** @see SubProcess
 ** @see CECEnum
 **/

public class Simplify {

    /***************************************************
     *                                                 *
     * Instance methods:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Our Simplify subprocess; no actions should be taken on this
     ** subprocess unless readySubProcess() is called first.
     **/
    private final /*@ non_null @*/ SubProcess P;

    /**
     ** This variable holds the CECEnum that is currently using
     ** Simplify (there can be at most 1 such CECEnum), or null if
     ** there is no such CECEnum. <p>
     **
     ** Simplify is available for use iff this is null.  Use
     ** readySubProcess() to make Simplify available. <p>
     **/
    private CECEnum subProcessUser = null;


    /***************************************************
     *                                                 *
     * Multiplexing Simplify:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Prepare Simplify for use. <p>
     **
     ** Ensures any CECEnum currently using Simplify finishes.<p>
     **
     ** Precondition: we are not closed.<p>
     **/
    private void readySubProcess() {
	if (subProcessUser!=null) {
	    subProcessUser.finishUsingSimplify();
	    eatPrompt();
	    subProcessUser = null;
	}
    }


    /***************************************************
     *                                                 *
     * Creation and destruction:		       *
     *                                                 *
     ***************************************************/

    /**
     ** Create a new invocation of Simplify as a sub-process. <p>
     **
     ** The resulting invocation is initially open, but may be closed
     ** permanently with the close method.<p>
     **
     ** This constructor may result in a fatal error if any problems
     ** occur.<p>
     **/
    public Simplify() {
	P = new SubProcess("Simplify",
			   java.lang.System.getProperty("simplify",
					"/usr/local/escjava/bin/Simplify"));
	eatPrompt();
    }


    /**
     ** Close us. <p>
     **
     ** This destroys the associated subprocess.  Afterwards, none of
     ** our methods should ever be called again.  Likewise, no methods
     ** of any Enumeration we've returned should ever be called
     ** again.<p>
     **
     ** This method may result in a fatal error if any problems
     ** occur.  Our subprocess is guaranteed to be destroyed afterwards,
     ** regardless of which exit is taken.<p>
     **/
    public void close() {
	P.close();
    }


    /***************************************************
     *                                                 *
     * Client operations:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Send a single <code>String</code> command to Simplify. <p>
     **
     ** A command is something that produces no output other than
     ** whitespace and a prompt.  If any other output is produced, 
     ** a fatal error will result.<p>
     **
     ** Precondition: we are not closed.<p>
     **/
    public void sendCommand(/*@ non_null @*/ String s) {
	readySubProcess();
	P.resetInfo();

	if (Info.on) {
	  Info.out("[calling Simplify with command '" + s + "']");
	}

	P.send(s);

	eatPrompt();

	Info.out("[Simplify: command done]");
    }


    /**
     ** Send a <code>String</code> containing zero-or-more commands to
     ** our Simplify subprocess. <p>
     **
     ** A command is something that produces no output other than
     ** whitespace and a prompt.  If any other output is produced, 
     ** a fatal error will result.<p>
     **
     ** Precondition: we are not closed.<p>
     **/
    public void sendCommands(/*@ non_null @*/ String s) {
	readySubProcess();
	P.resetInfo();

	if (Info.on) {
	  Info.out("[calling Simplify with commands '" + s + "']");
	}

	P.send(s);
	/*
	// this is so we can be sure we have resynchronized the stream:
	P.send(" (LBL |<resync>| (EQ |<A>| |<B>|))");

	// eat resulting prompts from s:
	while (P.peekChar() == '>') 
	    eatPrompt();

	// Eat output from synchronization expression:
	Enumeration results = prove("");
	try {
	  while (true) {
	    if (!results.hasMoreElements()) {
	      // get into exception handling code below
	      throw new SExpTypeError();
	    }
	    SimplifyOutput so = (SimplifyOutput)results.nextElement();
	    if (so.getKind() == SimplifyOutput.COUNTEREXAMPLE) {
	      SimplifyResult sr = (SimplifyResult)so;
	      if (sr.labels.at(0).getAtom().toString().equals("<resync>")) {
		// all is well (any remaining results will be eaten
		// with the next call to readySubProcess)
		break;
	      }
	    }
	  }
	} catch (Exception e) {
	    P.handleUnexpected("to have seen the label <resync>");
	}
	*/

	Info.out("[Simplify: commands done]");
    }


    /**
     ** Attempt to verify expression <code>exp</code>. <p>
     **
     ** We always return an Enumeration of non-null SimplifyOutput's.
     ** If verifying <code>exp</code> succeeds, the resulting Enumeration
     ** will end with a SimplifyOutput of kind VALID. <p>
     **
     ** Precondition: we are not closed, and <code>exp</code> is properly
     **		      formed according to Simplify's rules.<p> 
     **/
    //@ ensures \result!=null
    //@ ensures \result.elementType == \type(SimplifyOutput);
    //@ ensures !\result.returnsNull
    public Enumeration prove(/*@ non_null @*/ String exp) {
	readySubProcess();
	subProcessUser = new CECEnum(P, exp);
	return subProcessUser;
    }

    public void startProve() {
	readySubProcess();
	subProcessUser = new CECEnum(P);
    }

    //@ ensures \result!=null
    //@ ensures \result.elementType == \type(SimplifyOutput);
    //@ ensures !\result.returnsNull
    public Enumeration streamProve() {
        P.ToStream().flush();
	while (P.peekChar() == '>') {
            eatPrompt();
	}
	return subProcessUser;
    }

    public PrintStream subProcessToStream() {
	return P.ToStream();
    }

    /***************************************************
     *                                                 *
     * Utility routines:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Consume the prompt from our Simplify subprocess.  If the next
     ** output characters are not exactly the prompt (possibly preceeded
     ** by some whitespace), a fatal error is reported.<p>
     **
     ** Precondition: we are not closed.<p>
     **/
    private void eatPrompt() {
	P.eatWhitespace();
	P.checkChar('>');
	P.checkChar('\t');
    }


    /***************************************************
     *                                                 *
     * Test methods:				       *
     *                                                 *
     ***************************************************/

    /**
     ** A simple test routine
     **/
    public static void main(String[] args) throws IOException {
	Info.on = true;		// Turn on verbose mode

	Simplify S = new Simplify();

	exhaust(S.prove("(EQ A B)"));

	exhaust(S.prove("(EQ A A)"));

	S.sendCommand("(BG_PUSH (EQ B A))");
	exhaust(S.prove("(EQ A B)"));
	S.sendCommand("(BG_POP)");

	exhaust(S.prove("(EQ A B)"));

	S.sendCommands("(BG_PUSH (EQ B A)) (BG_POP)");

	// The the following is not a command and thus should produce an
	// error:
	try {
	    S.sendCommand("(EQ A A)");
	} finally {
	    S.close();
	}
    }


    /**
     ** Force an Enumeration to compute all of its elements
     **/
    static void exhaust(Enumeration n) {
	while (n.hasMoreElements())
	    n.nextElement();
    }
}
