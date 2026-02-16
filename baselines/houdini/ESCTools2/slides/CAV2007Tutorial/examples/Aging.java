public class Aging {

  private /*@ spec_public @*/ int age;

  /*@   requires 0 <= a && a <= 150;
    @   assignable age;
    @   ensures age == a;
    @ also
    @   requires a < 0;
    @   assignable \nothing;
    @   ensures age == \old(age);
    @*/
  public void setAge(int a)
  { if (0 <= a && a <= 150) { age = a; } }
}
