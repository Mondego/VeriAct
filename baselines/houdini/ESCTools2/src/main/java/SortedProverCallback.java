package escjava.sortedProver;

import escjava.sortedProver.NodeBuilder.SPred;

public interface SortedProverCallback
{
	/**
	 * Called when the prover finds a counter example, wants to inform about progress,
	 * error or other condition.
	 * 
	 * @param resp is the response to be processed, some responses provide methods to
	 * return information to the prover
	 */
	void processResponse(SortedProverResponse resp);
}
