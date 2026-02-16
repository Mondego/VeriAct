public class Animal implements Gendered {
  protected boolean gen; //@ in gender;
  /*@ protected represents 
    @     gender <- (gen ? "female" : "male");
    @*/

  protected /*@ spec_public @*/ int age = 0;

  /*@ requires g.equals("female")
    @       || g.equals("male");
    @ assignable gender;
    @ ensures gender.equals(g);  @*/
  public Animal(final String g)
  { gen = g.equals("female"); }

  public /*@ pure @*/ boolean isFemale() {
    return gen;
  }

  /*@   requires 0 <= a && a <= 150;
    @   assignable age;
    @   ensures age == a;
    @ also
    @   requires a < 0;
    @   assignable age;
    @   ensures age == \old(age);   @*/
  public void setAge(final int a)
  { if (0 <= a) { age = a; } }
}
