package escjava.sortedProver.simplify;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.channels.FileChannel;
import java.util.Date;
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
import escjava.sortedProver.CounterExampleResponse;
import escjava.sortedProver.EscNodeBuilder;
import escjava.sortedProver.SortedProver;
import escjava.sortedProver.NodeBuilder.SPred;
import escjava.sortedProver.SortedProverCallback;
import escjava.sortedProver.simplify.SimplifyNodeBuilder.Sx;
import escjava.sortedProver.NodeBuilder;
import escjava.translate.VcToString;

/*@ non_null_by_default @*/
public class SimplifyProver extends SortedProver
{
	private SimplifyNodeBuilder nodeBuilder;
	private SimplifyProcess simpl;
	private int pushHeight;
	private BackPred backPred;
	
	private SortedProverResponse ok;
  private SortedProverResponse yes;
  private SortedProverResponse no;
  private SortedProverResponse fail;

    // dump query to file with this extension
    private String optDumpExt;
    // when true, append the current time stamp to the dumped file
    // (before the extension)
    private boolean optDumpUnique = false;
    // dump all queries to this file
    private String dumpLog;
    // stream for dumpLog
    private PrintStream dumpStream;

  public SimplifyProver() {
    this(new String[] {System.getProperty("simplify", "simplify")});
  }

  public SimplifyProver(String[] cmd) {
    nodeBuilder = new SimplifyNodeBuilder();
    backPred = new BackPred();
    ok = new SortedProverResponse(SortedProverResponse.OK);
    yes = new SortedProverResponse(SortedProverResponse.YES);
    no = new SortedProverResponse(SortedProverResponse.NO);
    fail = new SortedProverResponse(SortedProverResponse.FAIL);
    optDumpExt = "";
    dumpLog = "simplify.log";
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
            try {
                optDumpUnique = Boolean.valueOf(properties.getProperty("DumpUnique",
                                                                          String.valueOf(optDumpUnique))).booleanValue();
            } catch (NumberFormatException e) {
                ErrorSet.fatal("Invalid input for option " + "DumpUnique" + " : " + e);
                return fail;
            }

	    optDumpExt = properties.getProperty("DumpExt", optDumpExt);
	    
	    return ok;
	}

	public SortedProverResponse sendBackgroundPredicate()	{
		backgroundPredicateSent = true;
		backPred.genUnivBackPred(simpl.out());
                if (optDumpExt != "") {
                    getDumpStream().flush();
                    backPred.genUnivBackPred(dumpStream);
                }
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
		String cmd = "(BG_PUSH\n" + formulaToString(formula) + "\n)";
		dump(cmd);
		simpl.sendCommand(cmd);
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
      String cmd = "(BG_PUSH\n" + formulaToString(formula) + "\n)";
      dump(cmd);
      simpl.sendCommand(cmd);
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
      while (count-- > 0) {
	String cmd = "(BG_POP)";
	dump(cmd);
	simpl.sendCommand(cmd);
      }
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

	    dump(form);
	    if (optDumpExt != "") {
		getDumpStream().flush();

		try {
		    FileChannel srcChannel = new FileInputStream(dumpLog).getChannel();
		    FileChannel dstChannel = new FileOutputStream(getDumpFile(properties)).getChannel();
		    dstChannel.transferFrom(srcChannel, 0, srcChannel.size());
		    srcChannel.close();
		    dstChannel.close();
		} catch (IOException e) {
		    ErrorSet.fatal("Couldn't copy dump file: " + dumpLog + " : " + e);
		}
	    }

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
		cleanupDumpStream();
		return ok;
	}


    private String quote(String name) {
        return name.replace('$', '_');
    }

    private String getDumpFile(Properties properties) {
        String unique = "";

        if (optDumpUnique) {
            unique = "_" + new Date().getTime();
        }

        String name = quote(properties.getProperty("ProblemName") + unique + "." + optDumpExt);

        return name;
    }

    public PrintStream getDumpStream() {
	if (dumpStream == null) {
	    try {
		dumpStream = new PrintStream(new FileOutputStream(dumpLog));
	    } catch (FileNotFoundException e) {
		ErrorSet.fatal("Couldn't create dump file: " + dumpLog + " : " + e);
		return null;
	    }
	}
	return dumpStream;
    }

    public void cleanupDumpStream () {
	if (dumpStream != null) {
	    dumpStream.close();
	    dumpStream = null;
	    (new File(dumpLog)).delete();
	}
    }

    public void dump(String cmd) {
	if (optDumpExt != "") {
	    getDumpStream().println(cmd);
	}
    }
}
