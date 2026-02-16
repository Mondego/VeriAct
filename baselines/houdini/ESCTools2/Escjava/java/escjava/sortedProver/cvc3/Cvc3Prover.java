package escjava.sortedProver.cvc3;

import java.nio.channels.FileChannel;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.Date;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Properties;

import javafe.util.Assert;
import javafe.util.ErrorSet;
import javafe.util.Info;

import escjava.Main;
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

import cvc3.*;

/*@ non_null_by_default @*/
public class Cvc3Prover extends SortedProver {

    //
    /// debug
    //
    final static private boolean verbose = false;
    final static private boolean printQuery = false; // verbose || true;

    // debug printing
    public static void print(Object o) {
	if (verbose) System.out.println(o);
    }


    // prover
    private cvc3.ValidityChecker prover;

    // expr builder
    private Cvc3NodeBuilder nodeBuilder;

    // lazy assertion of axioms
    // failed attempt to introduce only the axioms needed for a specific query -
    // in practice almost always all axioms were needed.
    //private Cvc3Axioms axiomManager;


    // possible prover responses
    final private SortedProverResponse ok = new SortedProverResponse(SortedProverResponse.OK);
    final private SortedProverResponse fail = new SortedProverResponse(SortedProverResponse.FAIL);
    final private SortedProverResponse yes = new SortedProverResponse(SortedProverResponse.YES);
    final private SortedProverResponse no = new SortedProverResponse(SortedProverResponse.NO);
    final private SortedProverResponse timeout = new SortedProverResponse(SortedProverResponse.TIMEOUT);


    // prover options
    // initialized in setProverResourceFlags

    // time limit in s - 0 is unlimited.
    // uses cvc3's soft timeout feature, i.e. it is not guaranteed
    // that the time limit is always obeyed.
    // the timeout is only set in isValid,
    // i.e. it only takes the check of the actual query into account,
    // not the preprocessing of the background predicate or the precondition.
    private int optTimeOut = 0;

    // :TODO: memory limit in MB - 0 is unlimited
    // currently ignored
    private int optMemOut = 1000;

    // max. number of counter examples to search for
    private int optCounterExamples = 1;

    // print the counterexample context in addition to returning the labels
    private boolean optPrintContext = false;

    // use manual triggers
    private boolean optManualTriggers = true;

    // use multitriggers
    private int optMaxIL = 999;

    // use multitriggers
    private boolean optMultiTriggers = true;

    // eagerly instantiate all triggers
    private boolean optInstAll = true;

    // rewrite nonnullelements
    private boolean optNonnullelements = true;

    // rewrite isAllocate
    private boolean optIsallocated = false;

    // use cvc3's builtin transitivity functionality
    private boolean optBuiltinTrans = true;

    // use the symbol classLiteral, or just drop it
    private boolean optUseClassLiteral = true;

    // use an inductive datatype to represent java types -
    // this is the major divergence from the original background predicate
    // as used by Simplify
    private boolean optUseDatatype = false;

    // use cvc preprocessing
    // enabling this might remove labeled expressions from the query,
    // thus making counterexample generation fail.
    private boolean optPreprocess = false;

    // cvc logs the query to this file,
    // i.e. it logs all queries send between startProver and stopProver.
    private String dumpLog = "cvc.log";
    // dump queries to file with this extension,
    // and with a name based on the current query,
    // i.e. class and method name.
    // each dump is a copy of dumpLog up to the current query,
    // i.e. it contains the previous queries as well.
    private String optDumpExt = "";
    // when true, append the current time stamp to the dumped file
    // (before the extension)
    // rational:
    // if EscJava is called on several files containing classes and methods
    // with overlapping names, getDumpFile will return the same name for those.
    // that is, dumping will in this case overwrite earlier vcs with newer ones.
    // this tends to happen quite a few times in the EscJava test suite
    // (Escjava/test/escjava/test/).
    private boolean optDumpUnique = false;


    // lazy assertion of axioms
    //private boolean optLazyAxiomatization = false;


    // generic flag for testing purposes
    private boolean optTest = false;


    public Cvc3Prover() {
	if (printQuery) System.out.println("%% Cvc3Prover :: constructor");
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


    /*
    private Cvc3Axioms getAxiomManager() {
	if (axiomManager == null) {
	    axiomManager = new Cvc3Axioms(optLazyAxiomatization);
	}
	return axiomManager;
    }
    */

    private cvc3.ValidityChecker getProver() {
	if (prover == null) {
	    if (printQuery) System.out.println("%% Cvc3Prover :: getProver");

	    // Need to have nodeBuilder available before startProver
	    // (due to usage in ProverManager), and need prover for nodeBuilder,
	    // so do initialization here instead of in startProver.
	    //
	    // Thus startProver as well as stopProver can only be called once
	    // for an instance of Cvc3Prover.
	    try {
		cvc3.FlagsMut flags = ValidityChecker.createFlags(null);
		// ? true can lead to crash in cvc3
		flags.setFlag("tcc", false);
		
		// soft timeout
                // setting the timeout here will impact cvc3's work occurring
                // before isValid is called, which is usually not what is wanted.
                //flags.setFlag("stimeout", (int) (9.8 * optTimeOut));

		// non-dags do sometimes take too much memory
		flags.setFlag("dagify-exprs", true);
		// make cvc log all queries to this prover instance
		if (optDumpExt != "") {
		//flags.setFlag("dump-trace", "cvc." + logCounter + ".trace");
		    flags.setFlag("dump-log", dumpLog);
                    // dumping may sometimes only works without dags
                    flags.setFlag("dagify-exprs", false);
		}
		
		//flags.setFlag("trace", "ALL", true);

                // need to disable preprocessing, so that cvc3 can be queried
                // for the truth values of (labeled) expressions
		flags.setFlag("preprocess", optPreprocess);
		// enable manual triggers
		flags.setFlag("quant-man-trig", optManualTriggers);
                // limit instantiation depth
		flags.setFlag("quant-max-IL", optMaxIL);
                // enable multi triggers
		//flags.setFlag("quant-inst-mult", optMultiTriggers);
		// ? true can lead to crash in cvc3
		//flags.setFlag("quant-polarity", false);
		// ? eagerly instantiate everything
		//flags.setFlag("quant-inst-all", optInstAll);
		//flags.setFlag("max-quant-inst", 400);

		flags.setFlag("trans-closure", optBuiltinTrans);

		prover = ValidityChecker.create(flags);
		flags.delete();

		nodeBuilder = new Cvc3NodeBuilder(prover, this,
                                                  optMultiTriggers, optManualTriggers,
                                                  optNonnullelements, optIsallocated, optBuiltinTrans,
                                                  optUseClassLiteral, optUseDatatype,
                                                  optTest);
	    } catch (Throwable e) {
		ErrorSet.fatal("Could not create cvc3: " + "e");
		// to silence javac complaining that prover might not be initialized,
		// as ErrorSet will throw an Error this will never be reached anyway:
		throw new Error("Could not create cvc3: " + "e");
	    }
	}
	Assert.notNull(prover);
	return prover;
    }

    public void printOptions() {
	System.out.println("%% Cvc3 Options:");
	System.out.println("%% TimeLimit      : " + optTimeOut);
	System.out.println("%% MemoryLimit    : " + optMemOut);
	System.out.println("%% CounterExamples: " + optCounterExamples);
	System.out.println("%% PrintContext   : " + optPrintContext);
	System.out.println("%% ManualTriggers : " + optManualTriggers);
	System.out.println("%% MaxIL : "          + optMaxIL);
	System.out.println("%% MultiTriggers  : " + optMultiTriggers);
	System.out.println("%% InstAll        : " + optInstAll);
	System.out.println("%% Nonnullelements: " + optNonnullelements);
	System.out.println("%% Isallocated    : " + optIsallocated);
	System.out.println("%% BuiltinTrans   : " + optBuiltinTrans);
	System.out.println("%% UseClassLiteral: " + optUseClassLiteral);
	System.out.println("%% UseDatatType   : " + optUseDatatype);
	System.out.println("%% Preprocess     : " + optPreprocess);
	//System.out.println("%% LazyAxioms     : " + optLazyAxiomatization);
	System.out.println("%% DumpExt        : " + optDumpExt);
	System.out.println("%% DumpUnique     : " + optDumpUnique);
	System.out.println("%% Test           : " + optTest);
    }

    public int stackLevel() {
	//	Assert.notFalse(started);
	try {
	    return prover.stackLevel();
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal("Could not get prover stackLevel: " + e);
	    return 0;
	}
    }

    public EscNodeBuilder getNodeBuilder() {
	if (nodeBuilder == null) {
	    Assert.notFalse(prover == null);
	    getProver();
	}
	Assert.notNull(nodeBuilder);
	return nodeBuilder;
    }

    public SortedProverResponse startProver() {
	if (printQuery) System.out.println("%% startProver: " + stackLevel());
	Assert.notNull(getProver());
	Assert.notFalse(!started);

	try {
	    started = true;
	    ((Cvc3NodeBuilder)getNodeBuilder()).setup();
	    return ok;
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal("Could not start cvc3: " + e);
	    return fail;
	}
    }

    public SortedProverResponse stopProver() {
	if (printQuery) System.out.println("%% stopProver: " + stackLevel());
	if (!started)
	  return ok;//Assert.notFalse(started);

	try {
	    started = false;
	    if (optDumpExt != "") {
		(new File(dumpLog)).delete();
	    }
	    prover.delete();
	    return ok;
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal("Could not stop cvc3: " + e);
	    return fail;
	}
    }

    public SortedProverResponse setProverResourceFlags(Properties properties) {
	if (printQuery) print("%% setProverResourceFlags");
	// :NOTE: this is called by ProverManager before startProver

	try {
	    optTimeOut = Integer.parseInt(properties.getProperty("TimeLimit", String.valueOf(optTimeOut)));
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "TimeLimit" + " : " + e);
	    return fail;
	}
	try {
	    optMemOut = Integer.parseInt(properties.getProperty("MemoryLimit", String.valueOf(optMemOut)));
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "MemoryLimit" + " : " + e);
	    return fail;
	}
	try {
	    optCounterExamples = Integer.parseInt(properties.getProperty("CounterExamples",
									 String.valueOf(optCounterExamples)));
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "CounterExamples" + " : " + e);
	    return fail;
	}
	try {
	    optPrintContext = Boolean.valueOf(properties.getProperty("PrintContext",
								     String.valueOf(optPrintContext))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "PrintContext" + " : " + e);
	    return fail;
	}
	try {
	    optManualTriggers = Boolean.valueOf(properties.getProperty("ManualTriggers",
								       String.valueOf(optManualTriggers))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "ManualTriggers" + " : " + e);
	    return fail;
	}
	try {
	    optMaxIL = Integer.parseInt(properties.getProperty("MaxIL",
								       String.valueOf(optMaxIL)));
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "MaxIL" + " : " + e);
	    return fail;
	}
	try {
	    optMultiTriggers = Boolean.valueOf(properties.getProperty("MultiTriggers",
								       String.valueOf(optMultiTriggers))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "MultiTriggers" + " : " + e);
	    return fail;
	}
	try {
	    optInstAll = Boolean.valueOf(properties.getProperty("InstAll",
								       String.valueOf(optInstAll))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "InstAll" + " : " + e);
	    return fail;
	}
	try {
	    optNonnullelements = Boolean.valueOf(properties.getProperty("Nonnullelements",
								       String.valueOf(optNonnullelements))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "Nonnullelements" + " : " + e);
	    return fail;
	}
	try {
	    optIsallocated = Boolean.valueOf(properties.getProperty("Isallocated",
								       String.valueOf(optIsallocated))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "Isallocate" + " : " + e);
	    return fail;
	}
	try {
	    optBuiltinTrans = Boolean.valueOf(properties.getProperty("BuiltinTrans",
								       String.valueOf(optBuiltinTrans))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "BuiltinTrans" + " : " + e);
	    return fail;
	}
	try {
	    optUseClassLiteral = Boolean.valueOf(properties.getProperty("UseClassLiteral",
								       String.valueOf(optUseClassLiteral))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "UseClassLiteral" + " : " + e);
	    return fail;
	}
	try {
	    optUseDatatype = Boolean.valueOf(properties.getProperty("UseDatatype",
								       String.valueOf(optUseDatatype))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "UseDatatype" + " : " + e);
	    return fail;
	}
	try {
	    optPreprocess = Boolean.valueOf(properties.getProperty("Preprocess",
								       String.valueOf(optPreprocess))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "Preprocess" + " : " + e);
	    return fail;
	}
	try {
	    optDumpUnique = Boolean.valueOf(properties.getProperty("DumpUnique",
								       String.valueOf(optDumpUnique))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "DumpUnique" + " : " + e);
	    return fail;
	}
	try {
	    optTest = Boolean.valueOf(properties.getProperty("Test",
								       String.valueOf(optTest))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "Test" + " : " + e);
	    return fail;
	}
	/*try {
	    optLazyAxiomatization = Boolean.valueOf(properties.getProperty("LazyAxioms",
								       String.valueOf(optLazyAxiomatization))).booleanValue();
	} catch (NumberFormatException e) {
	    ErrorSet.fatal("Invalid input for option " + "LazyAxioms" + " : " + e);
	    return fail;
            }*/
	
	optDumpExt = properties.getProperty("DumpExt", optDumpExt);

	if (printQuery) printOptions();

	return ok;
    }
	
    public SortedProverResponse sendBackgroundPredicate() {
	if (printQuery) System.out.println("%% sendBackgroundPredicate: " + stackLevel());
	Assert.notFalse(started);
	Assert.notFalse(!backgroundPredicateSent);
	try {
	    //nodeBuilder.sendBackgroundPredicate(this);
	    Cvc3BackgroundPredicate.sendBackgroundPredicate(this, nodeBuilder);
	    backgroundPredicateSent = true;
	    prover.push();
	    return ok;
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal("Could not send background predicate: " + e);
	    return fail;
	}
    }

    public SortedProverResponse declareAxiom(SPred formula) {
	if (printQuery) System.out.println("%% declareAxiom");
	Assert.notFalse(started);
	Assert.notFalse(stackLevel() == 0);
	try {
	    Cvc3Pred p = (Cvc3Pred) formula;
	    if (printQuery) {
		System.out.println("ASSERT (" + p.getExpr().toString() + ");");
	    }
	    //getAxiomManager().register(prover, p.getExpr());
	    prover.assertFormula(p.getExpr());
	    return ok;
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal("Could not declare axiom: " + e);
	    return fail;
	}
    }
	
    public SortedProverResponse makeAssumption(SPred formula) {	
	if (printQuery) System.out.println("%% makeAssumption");
	Assert.notFalse(started);
	try {
	    prover.push();
	    Cvc3Pred p = (Cvc3Pred) formula;
	    if (printQuery) p.getExpr().print(false);
	    //getAxiomManager().register(prover, p.getExpr());
	    prover.assertFormula(p.getExpr());
	    return ok;
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal("Could not declare axiom: " + e);
	    return fail;
	}
    }

    public SortedProverResponse retractAssumption(int count) {	    
	if (printQuery) System.out.println("%% retractAssumption: " + stackLevel() + " -> " + count);
	Assert.notFalse(started);
	Assert.notFalse(stackLevel() >= count);
	try {
	    prover.popTo(stackLevel() - count);
	    return ok;
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal("Could not retract assumptions: " + e);
	    return fail;
	}
    }


    public SortedProverResponse isValid(SPred formula, SortedProverCallback callback, Properties properties) {
	if (printQuery) System.out.println("%% isValid: " + stackLevel());
	Assert.notFalse(started);

	try {
	    if (printQuery) System.out.println("%% Query:");
	    Cvc3Pred vc = (Cvc3Pred) formula;

            vc = ((Cvc3NodeBuilder)getNodeBuilder()).guard(vc);

	    //if (printQuery) System.out.println(vc.getExpr());
	    if (printQuery) vc.getExpr().print(false);

	    prover.push();

	    //getAxiomManager().assertRequired(prover, vc.getExpr());
            // ms to s, with some gap for the soft timeout to be effective
            prover.setTimeLimit((int) (9.8 * optTimeOut));
	    cvc3.QueryResult result = prover.query(vc.getExpr());
            prover.setTimeLimit(0);

	    // copy current cvc3 log to dumpfile
	    if (optDumpExt != "") {
		try {
		    FileChannel srcChannel = new FileInputStream(dumpLog).getChannel();
                    //System.out.println(getDumpFile(properties));
                    //assert (!(new File(getDumpFile(properties))).exists());
		    FileChannel dstChannel = new FileOutputStream(getDumpFile(properties)).getChannel();
		    dstChannel.transferFrom(srcChannel, 0, srcChannel.size());
		    srcChannel.close();
		    dstChannel.close();
		} catch (IOException e) {
		    ErrorSet.fatal("Couldn't copy dump file: " + dumpLog + " : " + e);
		}
                // dump only, not interested in actually processing the query
		//if (result != QueryResult.VALID) {
		//    prover.pop();
		//}
		//return no;
	    }

	    if (result == QueryResult.INVALID //:TODO: check if same labels again if unknown also considered ok
		||
		result == QueryResult.UNKNOWN
		) {
		if (printQuery) {
		    if (result == QueryResult.INVALID) {
			System.out.println("STATUS: INVALID");
		    } else {
			System.out.println("STATUS: UNKNOWN");
		    }
		}

		Cvc3Labels labels = new Cvc3Labels(prover);
		labels.computeLabelContexts(vc);

		int numOfCounterExamples = 0;
		while ((numOfCounterExamples < optCounterExamples)
		        &&
		       (result == QueryResult.INVALID
			||
			result == QueryResult.UNKNOWN
			)) {
		    ++numOfCounterExamples;

		    Iterator i;

		    // print counterexample context
		    if (optPrintContext || verbose) {
			List counterExample = prover.getCounterExample();
			//HashMap cm = prover.getConcreteModel();
			System.out.println();
			System.out.println("Context:");
			i = counterExample.iterator();
			while (i.hasNext()) {
			    System.out.println(i.next().toString());
			}
			System.out.println();
		    }



		    List relevant = new ArrayList();
                    if (!optPreprocess) relevant = labels.relevantLabels();

		    List labelsExprs = new ArrayList();
		    i = relevant.iterator();
		    while (i.hasNext()) {
			String labelName = (String) i.next();
			if (!Cvc3Label.isMajor(labelName)) continue;

			Iterator j = labels.getLabeled(labelName).iterator();
			List labelExprs = new ArrayList();
			while (j.hasNext()) {
			    Cvc3Label label = (Cvc3Label) j.next();
			    if (label.isPos()) {
				labelExprs.add(label.getExpr());
			    } else {
				labelExprs.add(prover.notExpr(label.getExpr()));
			    }
			}
			labelsExprs.add(prover.orExpr(labelExprs));
		    }

		    // print relevant labels
		    i = relevant.iterator();
		    if (printQuery) {
			System.out.println();
			System.out.println("Relevant labels:");
			while (i.hasNext()) {
			    System.out.println(i.next());
			}
			System.out.println();
		    }

		    // print concrete model
		    //i = cm.entrySet().iterator();
		    //while (i.hasNext()) {
		    //  System.out.println(((Map.Entry)i.next()));
		    //}
		    
		    // counterexample is only relevant if a major label is relevant
		    if (labelsExprs.size() > 0) {
			String[] response = new String[relevant.size()];
			for (int j = 0; j < relevant.size(); ++j) {
			    response[j] = (String)relevant.get(j);
			}
                        
			//:TODO: pass context
			callback.processResponse(new CounterExampleResponse(response));

                        // look for next counterexample
                        if (numOfCounterExamples < optCounterExamples) {
                            Expr excludeExample = prover.notExpr(prover.andExpr(labelsExprs));
                            if (printQuery) System.out.println("Restart: " + excludeExample);
                            result = prover.restart(excludeExample);
                        }
		    } else {
			// no more relevant major labels
			break;
		    }
		}
		prover.pop();

		if (result == QueryResult.VALID) {
		    if (printQuery) System.out.println("STATUS: VALID");
		} else if (result == QueryResult.UNKNOWN) {
		    if (printQuery) System.out.println("STATUS: UNKNOWN");
		}

		return no;

	    } else if (result == QueryResult.VALID) {
		//:TODO: cvc pops automatically on unsat,
		// but what does EscJava expect?
		// will it call retractAssumption instead?
		prover.pop();
		if (printQuery) System.out.println("VALID");
		return yes;

	    } else if (result == QueryResult.UNKNOWN) {
		prover.pop();
		if (printQuery) System.out.println("UNKNOWN");
		return no;
	    } else {
		prover.pop();
		if (printQuery) System.out.println("???");
                Assert.notFalse(false);
		return fail;
	    } 
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal("Is valid failed: " + e);
	    return fail;
	}
    }
}
