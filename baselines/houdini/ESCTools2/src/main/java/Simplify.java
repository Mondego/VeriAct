/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.prover;

import java.io.*;
import java.util.Enumeration;

import javafe.util.*;

/**
 * <p> The interface to the Simplify theorem prover program using
 * Strings to send and S-expressions to receive. </p>
 *
 * <p> Note that a Java application will not exit normally (i.e., when
 * its <code>main()</code> method returns) until all subprocesses have
 * finished.  In particular, this implies that all
 * <code>Simplify</code> instances must be closed before this can
 * occur.  Alternatively, a Java application can just call
 * <code>java.lang.System.exit</code> to force an exit to occur
 * without waiting for subprocesses. </p>
 *
 *
 * <p> <strong>Warning:</strong> This class uses {@link
 * SubProcess#readSExp()}, which does not implement the full grammer
 * for SExps.  See that routine for the exact limitations. </p>
 *
 * @see SExp
 * @see SubProcess
 * @see escjava.prover.CECEnum
 */

public class Simplify
{
    /**
     * Our Simplify subprocess; no actions should be taken on this
     * subprocess unless {@link #readySubProcess()} is called first.
     */
    //@ spec_public
    private final /*@nullable*/ SubProcess P;

    //@ public model boolean closed;
    //@ represents closed <- (P == null);

    /**
     * This variable holds the {@link CECEnum} that is currently using
     * Simplify (there can be at most 1 such CECEnum), or
     * <code>null</code> if there is no such CECEnum.
     *
     * <p> Simplify is available for use iff this is
     * <code>null</code>.  Use {@link #readySubProcess()} to make
     * Simplify available. </p>
     */
    //@ spec_public
    private /*@nullable*/ CECEnum subProcessUser = null;


    // Multiplexing Simplify

    /**
     * Prepare Simplify for use.
     * <p> Precondition: we are not closed. </p>
     * <p> Ensures any {@link CECEnum} currently using Simplify
     * finishes. </p>
     */
    //@ requires !closed;
    //@ ensures subProcessUser != null ==> subProcessUser == null;
	private void readySubProcess() {
		if (subProcessUser != null) {
			subProcessUser.finishUsingSimplify();
			eatPrompt();
			subProcessUser = null;
		}
	}


    // Creation and destruction

    /**
     * Create a new invocation of Simplify as a sub-process.
     *
     * <p> The resulting invocation is initially open, but may be
     * closed permanently with the {@link #close()} method. </p>
     *
     * <p> This constructor may result in a fatal error if any
     * problems occur. </p>
     */
    //@ ensures !closed;
    public Simplify() {
	P = new SubProcess("Simplify",
                           new String[] {
                               java.lang.System.getProperty("simplify",
                                                            "/usr/local/escjava/bin/Simplify"),
			       "-noprune",
                               "-noplunge"}, // FIXME - make controllable
                           null);
	eatPrompt();
    }

    /**
     * Close us.
     *
     * <p> This destroys the associated subprocess.  Afterwards, none
     * of our methods should ever be called again.  Likewise, no
     * methods of any {@link Enumeration} we've returned should ever
     * be called again. </p>
     *
     * <p> This method may result in a fatal error if any problems
     * occur.  Our subprocess is guaranteed to be destroyed afterwards,
     * regardless of which exit is taken. </p>
     */
    //@ ensures closed;
    public void close() {
	P.close();
    }


    // Client operations

    /**
     * Send a single <code>String</code> command to Simplify.
     *
     * <p> A command is something that produces no output other than
     * whitespace and a prompt.  If any other output is produced, a
     * fatal error will result. </p>
     *
     * <p> Precondition: we are not closed. </p>
     *
     * @param s the string containing the command to be sent to
     * simplify.
     */
    //@ requires !closed;
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
     * Send a <code>String</code> containing zero-or-more commands to
     * our Simplify subprocess.
     *
     * <p> A command is something that produces no output other than
     * whitespace and a prompt.  If any other output is produced, a
     * fatal error will result. </p>
     *
     * <p> Precondition: we are not closed. </p>
     *
     * @param s the string containing the commands to be sent to
     * simplify.
     */
    //@ requires !closed;
    public void sendCommands(/*@ non_null @*/ String s) {
	readySubProcess();
	P.resetInfo();

	if (Info.on) {
            Info.out("[calling Simplify with commands '" + s + "']");
	}

	P.send(s);

        // review kiniry 14 March 2004 - Why is this code block no
        // longer necessary?

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
     * Attempt to verify expression <code>exp</code>.
     *
     * <p> We always return an {@link Enumeration} of
     * non-<code>null</code> {@link SimplifyOutput}s.  If verifying
     * <code>exp</code> succeeds, the resulting {@link Enumeration}
     * will end with a {@link SimplifyOutput} of kind {@link
     * SimplifyOutput#VALID}. </p>
     *
     * <p> Precondition: we are not closed, and <code>exp</code> is
     * properly formed according to Simplify's rules. </p>
     */
    //@ requires !closed;
    //@ ensures \result.elementType == \type(SimplifyOutput);
    //@ ensures !\result.returnsNull;
    public /*@ non_null @*/ Enumeration prove(/*@ non_null @*/ String exp) {
	readySubProcess();
	subProcessUser = new CECEnum(P, exp);
	return subProcessUser;
    }

    /**
     * Readies the simplify subprocess and sets its user to this, but
     * send no commands.
     */
    //@ normal_behavior
    //@   modifies subProcessUser;
    //@   ensures subProcessUser != null;
    public void startProve() {
	readySubProcess();
	subProcessUser = new CECEnum(P);
    }

    //@ ensures \result.elementType == \type(SimplifyOutput);
    //@ ensures !\result.returnsNull;
    public/* @ non_null @ */Enumeration streamProve() {
    	P.ToStream().flush();
    	while (P.peekChar() == '>') {
    		eatPrompt();
    	}
    	return subProcessUser;
    }

    //@ ensures \result == null <==> closed;
	public PrintStream subProcessToStream() {
		return P.ToStream();
	}

    
    // Utility routines

    /**
     * Consume the prompt from our Simplify subprocess.  If the next
     * output characters are not exactly the prompt (possibly preceeded
     * by some whitespace), a fatal error is reported.
     *
     * <p> Precondition: we are not closed. </p>
     */
    //@ requires !closed;
    //@ requires P != null;
    private void eatPrompt() {
	P.eatWhitespace();
	P.checkChar('>');
	P.checkChar('\t');
    }


    // Test methods

    /**
     * A simple test routine
     */
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

	// The following is not a command and thus should produce
	// an error:
	try {
	    S.sendCommand("(EQ A A)");
	} finally {
	    S.close();
	}
    }

    /**
     * Force an {@link Enumeration} to compute all of its elements
     */
    static void exhaust(/*@ non_null @*/ Enumeration n) {
	while (n.hasMoreElements())
	    n.nextElement();
    }
}
