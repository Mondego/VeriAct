package escjava.sortedProver.auflia;

import java.io.BufferedOutputStream;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Enumeration;
import java.util.Properties;
import java.util.Stack;

import javafe.util.Assert;
import javafe.util.ErrorSet;
import javafe.util.FatalError;
import javafe.util.Info;

import escjava.backpred.BackPred;
import escjava.sortedProver.SortedProverResponse;
import escjava.prover.SExp;
import escjava.prover.SList;
import escjava.prover.Simplify;
import escjava.prover.SimplifyOutput;
import escjava.prover.SimplifyResult;
import escjava.prover.SubProcess;
import escjava.prover.SubProcess.Died;
import escjava.sortedProver.CounterExampleResponse;
import escjava.sortedProver.EscNodeBuilder;
import escjava.sortedProver.NodeBuilder;
import escjava.sortedProver.SortedProver;
import escjava.sortedProver.SortedProverCallback;
import escjava.sortedProver.NodeBuilder.SPred;
import escjava.sortedProver.auflia.AufliaNodeBuilder.Sx;
import escjava.translate.VcToString;

/*@ non_null_by_default @*/
public class AufliaProver extends SortedProver
{
	protected Properties currentProperties = new Properties(); 
	
	protected AufliaNodeBuilder nodeBuilder = new AufliaNodeBuilder();
	int pushHeight;
	BackPred backPred = new BackPred();
	
	SortedProverResponse ok = new SortedProverResponse(SortedProverResponse.OK);

	public EscNodeBuilder getNodeBuilder()
	{
		return nodeBuilder;
	}

	public SortedProverResponse startProver()
	{
		started = true;
		return ok;
	}

	public SortedProverResponse setProverResourceFlags(Properties properties)
	{
		currentProperties.putAll(properties);
		return ok;
	}
	
	public SortedProverResponse sendBackgroundPredicate()
	{
		backgroundPredicateSent = true;		
		return ok;
	}
	
	protected Stack assumptions = new Stack(); 

	public SortedProverResponse declareAxiom(SPred formula) 
	{
		Assert.notFalse(pushHeight == 0);
		assumptions.push(formulaToString(formula));
		return ok;
	}
	
	String formulaToString(SPred form)
	{
		Sx formula = (Sx)form;
		StringBuffer sb = new StringBuffer();
		formula.dump(0, sb);		
		return sb.toString();
	}
	
	static class State
	{
		final public String formula;
		final public String extrafuns;
		
		public State(String f, String e)
		{
			formula = f;
			extrafuns = e;
		}
	}

	public SortedProverResponse makeAssumption(SPred formula)
	{	
		pushHeight++;
		String form = formulaToString(formula);
		String e = nodeBuilder.extrafuns.toString();
		nodeBuilder.extrafuns = new StringBuffer();
		assumptions.push(new State(form, e));
		return ok;
	}

	public SortedProverResponse retractAssumption(int count)
	{	    
		Assert.notFalse(pushHeight >= count);
		pushHeight -= count;
		while (count-- > 0)
			assumptions.pop();
		return ok;
	}
	
	protected void saveQuery(String filename, SPred formula)
	{
	    try {
	    	BufferedWriter f = new BufferedWriter(new FileWriter(filename));
	    	
	    	f.write(AufliaPrelude.prelude);
	    	for (int i = 0; i < assumptions.size(); ++i) {
	    		State s = (State)assumptions.get(i);
	    		f.write(s.extrafuns);
	    		f.write(":assumption ");
	    		f.write(s.formula);
	    		f.write("\n");
	    	}
	    	
	    	String form = formulaToString(nodeBuilder.buildNot(formula)); 
			String e = nodeBuilder.extrafuns.toString();
			nodeBuilder.extrafuns = new StringBuffer();
	    	f.write(e);
	    	
	    	formula = nodeBuilder.buildNot(formula);
	    	f.write(":formula\n");
	    	f.write(form);
	    	f.write("\n)\n");
	    	f.close();
	    } catch (IOException e) {
	    	ErrorSet.fatal("error writing the formula " + e);
	    }
	}
	
	public SortedProverResponse isValid(SPred formula, SortedProverCallback callback, Properties properties)
	{
		setProverResourceFlags(properties);
		String filename = encodeProblemName(properties) + ".smt"; 
	    saveQuery(filename, formula);	    
	    ErrorSet.caution("wrote formula to: " + filename + ", not proving anything! "); 
    	return new SortedProverResponse(SortedProverResponse.YES);
	}

	public SortedProverResponse stopProver()
	{
		started = false;
		return ok;
	}
}
