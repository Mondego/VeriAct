package escjava.sortedProver;

import escjava.sortedProver.NodeBuilder.SPred;

/**
 * This response is returned when the prover finds a counter example.
 */
/*@ non_null_by_default @*/
public class CounterExampleResponse extends SortedProverResponse
{
	private final String[] labels;
	
	/**
	 * Construct new instance.
	 *   
	 * @param labels the current set of labels describing the counter example
	 */
	//@ requires \nonnullelements(labels);
	public CounterExampleResponse(String[] labels)
	{
		super(COUNTER_EXAMPLE, 0);
		this.labels = labels;
	}
	
	//@ ensures \nonnullelements(\result);
	/*@ pure @*/public String[] getLabels()
	{
		return labels;
	}

	
	/**
	 * Can be called by the reciver of the response to discharge it.
	 * 
	 * @param pred The reciver can pass here a formula that would be conjoined to
	 * the original query to hint the direction in which 
	 * prover should look for further counter examples.  The prover can treat 
	 * the formula returned as a theory tautology (in
	 * which case it can say the formula is valid).  It is however free to ignore that.
	 */
	public void discharge(SPred pred)
	{		
	}
}
