package escjava.sortedProver.fx7;

import java.util.Enumeration;
import java.util.Properties;

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
import escjava.sortedProver.simplify.SimplifyNodeBuilder.Sx;
import escjava.translate.VcToString;

/*@ non_null_by_default @*/
public class Fx7Prover extends SortedProver
{
	Fx7NodeBuilder nodeBuilder = new Fx7NodeBuilder();
	int pushHeight;
	BackPred backPred = new BackPred();
	SubProcess fx7 = new SubProcess("fx7", new String[] { "fx7", "-simplify-syntax", "-mechanical", "-t:60" }, 
						null);
	
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
		Enumeration e = properties.keys();
		while (e.hasMoreElements()) {
			String key = (String)e.nextElement();
			if (key.equals("ProblemName")) continue;
			send("(PRAGMA " + key + " " + properties.getProperty(key) + ")\n");			
		}
		return ok;
	}
	
	String readLine()
	{
		fx7.eatWhitespace();
		String line = fx7.readWord("\n");
		if (line == "" && fx7.peekChar() < 0)
			line = "EOF";
		if (Info.on)
			Info.out("fx7: " + line);
		return line;
	}
	
	static final int INFO = 0;
	static final int WARN = 1;
	static final int ANSWER = 2;
	static final int ERROR = 3;
	static final int EOF = 4;
	
	int severity(String msg)
	{
		if (msg.startsWith("TEMP:") || msg.startsWith("INFO:") || msg.startsWith("GRIND:"))
			return INFO;
		else if (msg.startsWith("WARN:"))
			return WARN;
		else if (msg.startsWith("ANSWER:"))
			return ANSWER;
		else if (msg == "EOF")
			return EOF;
		else
			return ERROR;
	}
	
	String readLine(int severity)
	{
		String l;
		
		do {
			l = readLine();
		} while (severity(l) < severity);
		
		return l; 
	}
	
	void send(String s)
	{
		if (Info.on) {
			Info.out("[calling fx7 with:\n");
			System.out.print(s);
			Info.out("]");
		}
		fx7.send(s);
	}

	public SortedProverResponse sendBackgroundPredicate()
	{
		backgroundPredicateSent = true;
		backPred.genUnivBackPred(fx7.ToStream());
		return ok;
	}

	public SortedProverResponse declareAxiom(SPred formula) 
	{
		Assert.notFalse(pushHeight == 0);
		send("(BG_PUSH\n" + formulaToString(formula) + "\n)");
		return ok;
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
		send("(BG_PUSH\n" + formulaToString(formula) + "\n)");
		return ok;
	}

	public SortedProverResponse retractAssumption(int count)
	{	    
		Assert.notFalse(pushHeight >= count);
		pushHeight -= count;
		while (count-- > 0)
			send("(BG_POP)");
		return ok;
	}
	
	public SortedProverResponse isValid(SPred formula, SortedProverCallback callback, Properties properties)
	{
		setProverResourceFlags(properties);
	    String form = formulaToString(formula);
	    send(form);
	    String ans = readLine(ANSWER);
	    if (ans.startsWith("ANSWER: SAT")) {
	    	ans = readLine(INFO);
	    	if (ans.startsWith("INFO: labels: ")) {
	    		ans = ans.substring(14);
	    		String[] labels = ans.split(" ");
	    		for (int i = 0; i < labels.length; ++i)
	    			if (labels[i].startsWith("|EvP@") || labels[i].startsWith("|EvN@")) {
	    				Info.out("lab: " + labels[i]);
	    				labels[i] = labels[i].substring(5, labels[i].length() - 1);
	    				Info.out("  --> " + labels[i]);
	    			}
	    		callback.processResponse(new CounterExampleResponse(labels));
	    	} else {
	    		// It happens if there were no labels in the query.
	    		// ErrorSet.fatal("no labels, no donut: " + ans);
	    	}
	    	
			return new SortedProverResponse(SortedProverResponse.NO);
	    } else if (ans.startsWith("ANSWER: UNSAT")) {
	    	return new SortedProverResponse(SortedProverResponse.YES);
	    } else if (ans.startsWith("ANSWER: TIMEOUT")) {
	    	return new SortedProverResponse(SortedProverResponse.TIMEOUT);
	    } else {
	    	try {
	    		send("(PING)");
	    		boolean ok = false;
	    		while (true) {
	    			String line = readLine();
	    			if (line.equals("EOF")) {
	    				break;
	    			} else if (line.startsWith("ANSWER: PONG")) {
	    				ok = true;
	    				break;
	    			} else {
	    				if (ans.length() < 10000)
	    					ans += line + "\n";
	    			}
	    		}
	    	} catch (FatalError e) {	    			    		
	    	}
	    	
	    	// TODO if ok is set, then there is probably no need to restart the prover
	    	
	    	ErrorSet.fatal("got some evil response from fx7: " + ans);
	    	return new SortedProverResponse(SortedProverResponse.FAIL);
	    }
	}

	public SortedProverResponse stopProver()
	{
		started = false;
		fx7.close();
		return ok;
	}
}
