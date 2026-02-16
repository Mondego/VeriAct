class BogusFrameConditions
{
  public int i;

  public void l() {
    //@ assume i == 0;
    s();
    //@ assert true == false;
  }

  public void m() {
    //@ assume i == 0;
    t();
    //@ assert true == false;
  }

  public void n() {
    //@ assume i == 0;
    u();
    //@ assert true == false;
  }

  public void o() {
    //@ assume i == 0;
    v();
    //@ assert true == false;
  }


  //@ requires i == 0;
  //@ ensures i == 1;
  public void s() {
    i++;
  }

  //@ requires i == 0;
  //@ assignable \everything;
  //@ ensures i == 1;
  public void t() {
    i++;
  }

  //@ requires i == 0;
  //@ assignable this.*;
  //@ ensures i == 1;
  public void u() {
    i++;
  }

  //@ requires i == 0;
  //@ assignable i;
  //@ ensures i == 1;
  public void v() {
    i++;
  }
}
