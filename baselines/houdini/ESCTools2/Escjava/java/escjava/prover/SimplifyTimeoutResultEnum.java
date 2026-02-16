package escjava.prover;

import java.util.Enumeration;

/**
 * This class emulates a result enum from Simplify indicating that it
 * has timed out.
 */
public class SimplifyTimeoutResultEnum implements Enumeration {

	private boolean elementRead = false;
	
	// Use UNKNOWN rather than EXCEEDED_PROVER_KILL_TIME
	// because it produces a "time out" warning whereas
	// the latter resulted in a "failed" status.
	private static final SimplifyResult timeoutResult = new SimplifyResult(SimplifyResult.UNKNOWN,null,null);

	public boolean hasMoreElements() {
		return !elementRead;
	}

	public Object nextElement() {
		elementRead = true;
		return timeoutResult;
	}

}
