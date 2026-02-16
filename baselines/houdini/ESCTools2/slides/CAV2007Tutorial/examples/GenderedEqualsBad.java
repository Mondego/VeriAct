public interface GenderedEqualsBad extends Gendered {
  /*@ also
    @   ensures obj instanceof Gendered
    @       ==> (\result
    @              == gender.equals(
    @                   ((Gendered)obj).gender));
    @*/
  public /*@ pure @*/ 
  boolean equals(/*@ nullable @*/ Object obj);
}
