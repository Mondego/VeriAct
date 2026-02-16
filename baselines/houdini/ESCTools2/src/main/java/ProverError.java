package escjava.sortedProver.simplify;


/**
 *  Signals a serious error that occurred while chatting with
 *  Simplify (or someone impersonating Simplify).
 */
public class ProverError extends Exception {
  public ProverError(String msg) {
    super(msg);
  }

  public ProverError(String msg, Exception w) {
    super(msg, w);
  }
}
