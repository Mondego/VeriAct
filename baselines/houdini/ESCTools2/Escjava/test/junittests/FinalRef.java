// This case show that final reference objects aren't presumed known.

public class FinalRef {
FinalRef();

final String empty = "";
final int i = 8;
final Integer j = new Integer(9);
final Integer jj = j;

public void m() {
    //@ assert i == 8;
}

public void mj() {
    //@ assert j.intValue() == 9;
}

public void mjj() {
    //@ assert jj == j;
}

public void mm() {
    //@ assert String.equals("",empty);
}

public void ms() {
    //@ assert empty == "";
}

//@ requires empty == "";
//@ requires j == new Integer(9);
public void n() {
    //@ assert i == 8;
    //@ assert j.intValue() == 9;
    //@ assert String.equals("",empty);
}

}
