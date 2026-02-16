public class Person extends Animal {
  protected /*@ spec_public @*/ boolean ageDiscount = false; //@ in age;

  /*@ also
    @   requires \same;
    @   assignable age, ageDiscount;
    @   ensures 65 <= age ==> ageDiscount;   @*/
  public void setAge(final int a) {
    super.setAge(a);
    if (65 <= age) { ageDiscount = true; }
  }

  //@ requires g.equals("female") || g.equals("male");
  //@ assignable gender;
  //@ ensures gender.equals(g);
  public Person(final String g) { super(g); }
}
