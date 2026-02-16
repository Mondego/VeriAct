package escjava.prover;

public class Signature {
  private String signature;

  Signature(String signature) {
    this.signature = signature;
  }

  public /*@non_null*/String toString() {
    return signature;
  }
}
