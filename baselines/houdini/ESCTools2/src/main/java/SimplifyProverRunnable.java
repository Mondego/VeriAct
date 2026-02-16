/* Copyright 2008, DSRG.org */

package escjava.prover;

import java.util.Enumeration;
import java.util.Vector;

import escjava.ProverManager;

public class SimplifyProverRunnable implements Runnable {

	private Simplify simplify;
	private /*@nullable*/ Enumeration proverResult = null;

	public SimplifyProverRunnable(Simplify simplify) {
		this.simplify = simplify;
	}

	public void run() {
		if (ProverManager.THREADING_TRACE_OUTPUT > 9) System.out.println("\tSimplifyProverRunnable: run: starting");
		this.proverResult = this.streamProve();
		if (ProverManager.THREADING_TRACE_OUTPUT > 9) System.out.println("\tSimplifyProverRunnable: run: returning");
	}

	private Enumeration streamProve() {
		return simplify.streamProve();
	}

	public Enumeration getProverResult() {
		return proverResult;
	}
}
