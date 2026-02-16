// $Id$

/**
 * A demonstration (stub) SMT-LIB prover to show how ESC/Java2
 * interfaces with a prover written in Java.
 *
 * @author Joseph Kiniry
 */

import java.util.Properties;

class DemonstrationProver implements ProverInterface
{
  private boolean prover_running; //@ in objectState;
  //@ private represents prover_started <- prover_running;
  //@ private represents prover_stopped <- !prover_running;

  private /* null @*/ Signature signature; //@ in objectState;
  //@ private represents signature_defined <- (signature != null);

  static {
    System.loadLibrary("demoprover");
  }

  // start_solver : unit -> int
  /*@ public normal_behavior
    @   assignable objectState;
    @*/
  public native int demo_start_prover();

  // @review kiniry - how do we put whatever a native method modifies
  // into the objectState datagroup?

  public ProverResponse start_prover() {
    prover_running = true;
    return ProverResponse.factory(demo_start_prover());
  }

  // set_flags : string -> int
  public native int demo_set_prover_resource_flags(String properties);

  public ProverResponse set_prover_resource_flags(Properties properties) {
    return ProverResponse.OK;
  }

  // decaration : string -> int
  public native int demo_signature(String signature);

  public ProverResponse signature(Signature signature) {
    this.signature = signature;
    return ProverResponse.OK;
  }

  // add_axiom : string -> int
  public native int demo_declare_axiom(String axiom);

  public ProverResponse declare_axiom(Formula formula) {
    return ProverResponse.OK;
  }

  // add_assertion : string -> int
  public native int demo_make_assumption(String formula);

  public ProverResponse make_assumption(Formula formula) {
    return ProverResponse.OK;
  }

  // backtrack : int -> int
  public native int demo_retract_assumption(int count);

  public ProverResponse retract_assumption(int count) {
    return ProverResponse.OK;
  }

  public ProverResponse retract_assumption() {
    return ProverResponse.OK;
  }

  // query : string -> int
  public native int demo_is_valid(String formula, String properties);

  public ProverResponse is_valid(Formula formula,
                                 Properties properties) {
    return ProverResponse.OK;
  }

  // stop_solver : unit -> int
  /*@ public normal_behavior
    @   assignable objectState;
    @*/
  public native int demo_stop_prover();

  public ProverResponse stop_prover() {
    prover_running = false;
    return ProverResponse.factory(demo_stop_prover());
  }

  // quick-and-dirty unit test for starting and stopping prover
  public static void main(String[] args) {
    DemonstrationProver prover = new DemonstrationProver();
    prover.start_prover();
    System.out.println("Prover started.");
    prover.stop_prover();
    System.out.println("Prover stopped.");
  }
}
