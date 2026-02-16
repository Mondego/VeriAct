package escjava.prover;

public class Formula {
  private String formula;

  Formula(String formula) {
    this.formula = formula;
  }

  public /*@non_null*/String toString() {
    return formula;
  }
}
