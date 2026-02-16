package escjava.sortedProver.cvc;

import java.util.Enumeration;
import java.util.Properties;

import javafe.util.Assert;
import javafe.util.ErrorSet;
import javafe.util.Info;

import escjava.backpred.BackPred;
import escjava.sortedProver.SortedProverResponse;
import escjava.prover.SExp;
import escjava.prover.SList;
import escjava.prover.Simplify;
import escjava.prover.SimplifyOutput;
import escjava.prover.SimplifyResult;
import escjava.prover.SubProcess;
import escjava.sortedProver.CounterExampleResponse;
import escjava.sortedProver.EscNodeBuilder;
import escjava.sortedProver.NodeBuilder;
import escjava.sortedProver.SortedProver;
import escjava.sortedProver.SortedProverCallback;
import escjava.sortedProver.NodeBuilder.SPred;
import escjava.sortedProver.simplify.SimplifyNodeBuilder.Sx;
import escjava.translate.VcToString;

/*@ non_null_by_default @*/
public class CvcProver extends SortedProver
{
    // textual interface to prover as subprocess
    final SubProcess prover = new SubProcess("cvc3",
					     new String[] {
						 "/media/data/ESCJava/cvc3/bin/cvc3",
						 "+interactive",
						 "+tcc",
						 "+trace", "all"
					     },
					     null); 

    // ?
    final CvcNodeBuilder nodeBuilder = new CvcNodeBuilder();

    // number of pushes / assertions to prover
    int pushHeight;

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
	return ok;
    }
	
    public SortedProverResponse sendBackgroundPredicate()
    {
	backgroundPredicateSent = true;
	//new BackPred().genUnivBackPred(prover.ToStream());
	return ok;
    }

    public SortedProverResponse declareAxiom(SPred formula) 
    {
	Assert.notFalse(pushHeight == 0);
	//send("ASSERT(" + formulaToString(formula) + ");\n");
	return ok;
    }
	
    String formulaToString(SPred form)
    {
	return "FALSE";
	/*
	Sx formula = (Sx)form;
	StringBuffer sb = new StringBuffer();
	formula.dump(0, sb);		
	return sb.toString();
	*/
    }

    public SortedProverResponse makeAssumption(SPred formula)
    {	
	pushHeight++;
	//	send("PUSH;\n)");
	//	send("ASSERT(" + formulaToString(formula) + ");\n");
	return ok;
    }

    public SortedProverResponse retractAssumption(int count)
    {	    
	Assert.notFalse(pushHeight >= count);
	pushHeight -= count;
	while (count-- > 0)
	    send("POP;\n");
	return ok;
    }
	
    public SortedProverResponse isValid(SPred formula, SortedProverCallback callback, Properties properties)
    {
	send("QUERY(\n" + formulaToString(formula) + ");\nCOUNTERMODEL;");
	while (true) {
	String ans = readLine(ANSWER);
	if (ans.startsWith("CVC> - Invalid.")) {
	    return new SortedProverResponse(SortedProverResponse.NO);
	} else if (ans.startsWith("unknown.")) {
	    return new SortedProverResponse(SortedProverResponse.NO);
	} else if (ans.startsWith("CVC> - Valid.")) {
	    return new SortedProverResponse(SortedProverResponse.YES);
	    //	} else {
	    //	    ErrorSet.fatal("unexpected response from cvc: " + ans);
	    //	    return new SortedProverResponse(SortedProverResponse.FAIL);
	}
	}
    }
    
    public SortedProverResponse stopProver()
    {
	started = false;
	prover.close();
	return ok;
    }




    static final int INFO = 0;
    static final int WARN = 1;
    static final int ANSWER = 2;
    static final int ERROR = 3;
    static final int EOF = 4;
    
    String readLine()
    {
	prover.eatWhitespace();
	String line = prover.readWord("\n");
	if (line == "" && prover.peekChar() < 0)
	    line = "EOF";
	if (Info.on)
	    Info.out("cvc: " + line);
	return line;
    }
    
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
	if (Info.on)
	    Info.out("[calling cvc with:\n" + s + "\n]");
	    prover.send(s);
    }

}
