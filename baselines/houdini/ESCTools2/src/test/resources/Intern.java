public class Intern {
Intern();

public void m() {
  String s = "one";
  String ss = "two";
  String sss = "one";
  //@ assert String.isInterned(s);
  //@ assert String.isInterned(ss);
  //@ assert !String.equals(s,ss);
  //@ assert String.equals(s,sss);
  //@ assert String.equals(s,"one");
  //@ assert s == sss;
  //@ assert s == "one";

  String s4 = new String("one");
  String s5 = s4.intern();

  //@ assert !String.isInterned(s4);
  //@ assert String.isInterned(s5);
  //@ assert s != s4;

  //@ assert String.equals(s,s4);
  //@ assert String.equals(s,s5);
  //@ assert !String.equals(ss,s4);
  //@ assert !String.equals(ss,s5);

  //@ assert s == s5;

  //@ assert false; // TEST FOR CONSISTENCY
}

}
