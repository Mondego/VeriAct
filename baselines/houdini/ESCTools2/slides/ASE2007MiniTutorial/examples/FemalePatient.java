public class FemalePatient extends Patient {
  //@ public invariant gender.equals("female");

  //@ assignable gender;
  public FemalePatient() { super("female"); }
}
