public class Intern2 {
Intern2();

public void m() {
  String s = "one";
  String s4 = new String("one");
  String s5 = s4.intern();
  //@ assert String.equals(s,s4);
  // @ assert !String.equals(s,s5); // SHOULD BE ERROR
  //@ assert String.equals(s,s5); // SHOULD BE OK
//@ assert false; // TEST FOR CONSISTENCY
}

}
