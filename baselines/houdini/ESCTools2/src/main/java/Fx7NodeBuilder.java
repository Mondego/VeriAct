package escjava.sortedProver.fx7;

import escjava.sortedProver.simplify.SimplifyNodeBuilder;

/*@ non_null_by_default @*/
public class Fx7NodeBuilder extends SimplifyNodeBuilder
{
	public SPred buildLabel(boolean positive, String name, SPred pred)
	{
		if (positive)
			return sx("AND", sx("EvP@" + name), pred);
		else
			return sx("OR", sx("EvN@" + name), pred);
	}
	
	protected String integralPrintName(long n)
	{
		return String.valueOf(n);
	}
}
