public interface GreekNoun extends Gendered {
  /*@ public instance invariant gender.equals("female")
    @              || gender.equals("male") || gender.equals("neuter");  @*/

  //@ ensures \result <==> gender.equals("male");
  /*@ pure @*/ boolean isMale();
}
