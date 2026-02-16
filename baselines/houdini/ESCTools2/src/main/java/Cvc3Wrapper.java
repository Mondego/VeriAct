package escjava.prover.jniw;

public class Cvc3Wrapper {
  private native int natStartSolver();
  private native int natStopSolver();
  private native int natDeclType(String s);
  private native int natDeclPreds(String s);
  private native int natDeclFuns(String s);
  private native int natAssert(String s);
  private native String natQuery(String s);
  private native int natUndo(int n);
  private native int natSetFlags(String s);


  /** 
   * Begins a new instance of CVC3.  It is necessary to
   * start a fresh instance both initially and after each call to
   * stopSolver.
   *
   * @throws Cvc3WrapperException if there is an error starting the
   * solver.  This is probably a critical error, and the system may not
   * be able to (re)start properly.
   */ 
  public void startSolver() throws Cvc3WrapperException {
    if (natStartSolver() != 0) {
      throw new Cvc3WrapperException("Error starting solver");
    }
  }

  /**
   * Destroys the current instance of CVC3.  It is necessary
   * re-start the solver in order to redefine any types, functions, or
   * predicates.
   *
   * @throws Cvc3WrapperException if there is any error stopping the
   * solver.  This is probably a critical error and the system may not
   * be able to re-start properly.
   */
  public void stopSolver() throws Cvc3WrapperException {
    if (natStopSolver() != 0) {
      throw new Cvc3WrapperException("Error stopping solver");
    }
  }

  /**
   * Declares a new type.  This may be either a new 
   * uninterpreted type, a new complex type (such as a record type), 
   * or an alias of an existing type.
   *
   * @param s the new type declaration.  This must be a cvc3-readable 
   * command string (ending in a semicolon for the presentation language)
   *
   * @throws Cvc3WrapperException if there are any errors.
   */
  public void declType(String s) throws Cvc3WrapperException {
    if (natDeclType(s) != 0) {
      throw new Cvc3WrapperException("Error declaring type "+s);
    }
  }

  /**
   * Declares a predicate's type.
   *
   * @param s the predicate(s) declaration.  This must be a cvc3-readable
   * command string (ending in a semicolon for the presentation language).
   * For the presentation language it is necessary to indicate the 
   * predicate is of type BOOLEAN.
   *
   * @throws Cvc3WrapperException if there are any errors.
   */
  public void declPred(String s) throws Cvc3WrapperException {
    if (natDeclPreds(s) != 0) {
      throw new Cvc3WrapperException("Error declaring pred "+s);
    }
  }

  /**
   * Declares a function's type.  Note unbound (non-predicate) constants are 
   * considered 0-ary functions.
   *
   * @param s the function(s) declaration.  This must be a cvc3-readable
   * command string (ending in a semicolon for the presentation language).
   *
   * @throws Cvc3WrapperException if there are any errors.
   */
  public void declFun(String s) throws Cvc3WrapperException {
    if (natDeclFuns(s) != 0) {
      throw new Cvc3WrapperException("Error declaring function "+s);
    }
  }

  /**
   * Asserts a formula.  This adds a formula to the current context in a
   * stack-like manner.  It is possible to undo assertions.
   *
   * NOTE: If there are
   * multiple commands in a call to assertFormula, this counts as one
   * undoable instance.
   *
   * @param s the formula to be added.  This must be a cvc3-readable
   * command string (beginning with ASSERT and ending in a semicolon 
   * for the presentation language).
   *
   * @throws Cvc3WrapperException if there are any errors.  In this case
   * the assertion does not "count" for undos.
   *
   * @see undoAssertion
   */
  public void assertFormula(String s) throws Cvc3WrapperException {
    if (natAssert(s) != 0) {
      throw new Cvc3WrapperException("Error asserting formula "+s);
    }
  }

  /**
   * Queries a formula.  This attempts to determine if a formula is valid
   * in the current context.
   *
   * @param s the formula to be queried.  This must be a cvc3-readable
   * command string (beginning with QUERY and ending in a semicolon 
   * for the presentation language).
   *
   * @return a string with one of the following contents:
   * <tt><ul>
   * <li> Don't know: <i>lits</i>
   * <li> Abort
   * <li> Valid
   * <li> Invalid: <i>lits</i>
   * </ul></tt>
   * where <tt><i>lits</i></tt> is a list of space-separated 
   * <strong>TRUE</strong> literals from any asserted formulas.  Each 
   * literal will be contained in parentheses, and negative literals will
   * be prefixed by a tilde.  For example: <tt>(a) (~b)</tt>
   * In the case of "Don't know", this may be a spurious counterexample.
   *
   * @throws Cvc3WrapperException if there are any errors.  In this case
   * the assertion does not "count" for undos.
   */
  public String queryFormula(String s) throws Cvc3WrapperException {
    String o = natQuery(s);
    if (o.indexOf("JNIW_query ERROR") == 0) {
      throw new Cvc3WrapperException(o);
    }
    return o;
  }

  /**
   * Undoes the last n assertions.
   *
   * @param n the number of assertions to undo.  If n &lt = 0, nothing is 
   * done.  If n &gt the number of assertions, all assertions will be 
   * undone.
   *
   * @return The current scope of the solver -- the number of assertions
   * still in the current context.  This is likely only useful
   * for debugging purposes.
   *
   * @see assertFormula
   */
  public int undoAssert(int n) {
    return natUndo(n);
  }

  /**
   * Sets one or more CVC3 command-line flags.  Some flags may only be set
   * before the solver is started.  Others may be set at any time.  If a
   * flag is attempted to be set at the incorrect time, then nothing will
   * change.  See the CVC3 documentation for more information.
   *
   * @param s the flag string.  This must use the cvc3 command line
   * parameter conventions.
   *
   * @throws Cvc3WrapperException if there is an unrecognized flag or error.
   */
  public void setFlags(String s) throws Cvc3WrapperException {
    int err = natSetFlags(s);
    switch (err) {
      case -1 : throw new Cvc3WrapperException("Unrecognized flag "+s);
      case -2 : throw new Cvc3WrapperException("Solver is not running");
      default:
    }
  }

  /**
   * Checks the inconsistency of the current context.
   * 
   * @return <tt>true</tt> if current context is inconsistent.
   */
  public boolean contextInconsistent() throws Cvc3WrapperException {
    if (queryFormula("QUERY FALSE;").equals("Valid")) {
      return true;
    } else {
      return false;
    }
  }


  static {
    System.loadLibrary("cvc3jniw");
  }
}
