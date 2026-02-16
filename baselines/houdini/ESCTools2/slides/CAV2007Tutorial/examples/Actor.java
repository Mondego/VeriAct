public class Actor {

  private /*@ spec_public @*/ int age;
  private /*@ spec_public @*/ int fate;

  //@ public invariant 0 <= age && age <= fate;

  /*@   public normal_behavior
    @     requires age < fate - 1;
    @     assignable age;
    @     ensures age == \old(age+1);
    @ also
    @   public exceptional_behavior
    @     requires age == fate - 1;
    @     assignable age;
    @     signals_only DeathException;
    @     signals (DeathException e)
    @              age == fate;
    @*/
  public void older()
    throws DeathException
  { if (age < fate-1) { age++; }
    else throw new DeathException(); }
}
