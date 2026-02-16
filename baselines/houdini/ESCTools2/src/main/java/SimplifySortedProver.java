package escjava.sortedProver.simplify;

import java.util.Enumeration;
import java.util.Properties;

import javafe.util.Assert;
import javafe.util.Info;

import escjava.backpred.BackPred;
import escjava.sortedProver.SortedProverResponse;
import escjava.sortedProver.CounterExampleResponse;
import escjava.sortedProver.EscNodeBuilder;
import escjava.sortedProver.SortedProver;
import escjava.sortedProver.NodeBuilder.SPred;
import escjava.sortedProver.SortedProverCallback;
import escjava.sortedProver.simplify.SimplifyNodeBuilder.Sx;
import escjava.sortedProver.NodeBuilder;
import escjava.translate.VcToString;

/*@ non_null_by_default @*/
public class SimplifySortedProver extends SortedProver
{
	private SimplifyNodeBuilder nodeBuilder;
	private SimplifyProcess simpl;
	private int pushHeight;
	private BackPred backPred;
	
	private SortedProverResponse ok;
  private SortedProverResponse yes;
  private SortedProverResponse no;
  private SortedProverResponse fail;

  public SimplifySortedProver() {
    this(new String[] {System.getProperty("simplify", "simplify")});
  }

  public SimplifySortedProver(String[] cmd) {
    nodeBuilder = new SimplifyNodeBuilder();
    backPred = new BackPred();
    ok = new SortedProverResponse(SortedProverResponse.OK);
    yes = new SortedProverResponse(SortedProverResponse.YES);
    no = new SortedProverResponse(SortedProverResponse.NO);
    fail = new SortedProverResponse(SortedProverResponse.FAIL);
    try {
      simpl = new SimplifyProcess(cmd);
    } catch (ProverError e) {
      started = false;
    }
  }

	public EscNodeBuilder getNodeBuilder() {
		return nodeBuilder;
	}

	public SortedProverResponse startProver()	{
		started = true;
		return ok;
	}

	public SortedProverResponse setProverResourceFlags(Properties properties)	{
		return ok;
	}

	public SortedProverResponse sendBackgroundPredicate()	{
		backgroundPredicateSent = true;
		backPred.genUnivBackPred(simpl.out());
    try {
  		simpl.sendCommand("");
	  	return ok;
    } catch (ProverError e) {
      started = false;
      return fail;
    }
	}

	public SortedProverResponse declareAxiom(SPred formula) 
	{
		Assert.notFalse(pushHeight == 0);
    try {
  		simpl.sendCommand("(BG_PUSH\n" + formulaToString(formula) + "\n)");
	  	return ok;
    } catch (ProverError e) {
      started = false;
      return fail;
    }
	}
	
	String formulaToString(SPred form)
	{
		Sx formula = (Sx)form;
		StringBuffer sb = new StringBuffer();
		formula.dump(0, sb);		
		return sb.toString();
	}

	public SortedProverResponse makeAssumption(SPred formula)
	{	
		pushHeight++;
    try {
      simpl.sendCommand("(BG_PUSH\n" + formulaToString(formula) + "\n)");
      return ok;
    } catch (ProverError e) {
      --pushHeight;
      started = false;
      return fail;
    }
	}

	public SortedProverResponse retractAssumption(int count)
	{
		Assert.notFalse(pushHeight >= count);
    try {
      pushHeight -= count;
      while (count-- > 0)
        simpl.sendCommand("(BG_POP)");
      return ok;
    } catch (ProverError e) {
      started = false;
      return fail;
    }
	}
	
	public SortedProverResponse isValid(SPred formula, SortedProverCallback callback, Properties properties)
	{
    try {
      String form = formulaToString(formula);
      if (Info.on)
        Info.out("[proving formula\n" + form + "]");
      boolean result = simpl.isValid(form);
      if (!result) 
        callback.processResponse(new CounterExampleResponse(simpl.getLabels()));
      return result ? yes : no;
    } catch (ProverError e) {
      started = false;
      return fail;
    }
	}

	public SortedProverResponse stopProver() {
		started = false;
		simpl.stopProver();
		return ok;
	}
}
