public interface Gendered {
  //@ public model instance String gender;

  //@ ensures \result == gender.equals("female");
  public /*@ pure @*/ boolean isFemale();
}
