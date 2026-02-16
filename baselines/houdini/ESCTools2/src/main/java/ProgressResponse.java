package escjava.sortedProver;

/**
 * This response is returned by the prover from time to time, the estimates are by no means accurate.
 */
/*@ non_null_by_default @*/
public class ProgressResponse extends SortedProverResponse
{
	protected int progress;
	protected int eta;
	
	/**
	 * New instance.  
	 * 
	 * @param progress should be number of things done (for example conflicts found)
	 * @param eta should be the estimated total number of things done
	 */
	public ProgressResponse(int progress, int eta)
	{
		super(PROGRESS_INFORMATION, 0);
		this.progress = progress;
		this.eta = eta;
	}
	
	/**
	 * Can be called by the reciver to tell the prover to stop trying to prove the formula.
	 * 
	 * @note not sure about this one, maybe it's better to have a method in SortedProver?
	 */
	public void stopTrying()
	{	
	}
	
	/*@ pure @*/public int getEta()
	{
		return eta;
	}
	
	/*@ pure @*/public int getProgress()
	{
		return progress;
	}
	
	/*@ pure @*/public double estimateProgress()
	{
		return (double)progress / eta;
	}
}
