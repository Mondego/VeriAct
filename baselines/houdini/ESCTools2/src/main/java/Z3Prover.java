package escjava.sortedProver.z3;

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

import escjava.sortedProver.simplify.SimplifyProver;

/*@ non_null_by_default @*/
public class Z3Prover extends SimplifyProver
{
  public Z3Prover() {
    super(new String[]{System.getProperty("z3", "z3"), "/si"});
  }
}
