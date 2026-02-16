// Tests that integral division and modulus behave correctly

public class IntegralDivMod {


  public void m(int i, int j) {
    //@ assume j != 0;
    //@ assert (i%j) + (i/j)*j == i;
  }
  public void ma(int i, int j) {
    //@ assume j != 0;
    //@ assert i>=0 ==> (i%j) >= 0;
    //@ assert i<=0 ==> (i%j) <= 0;
  }
  public void mb(int i, int j) {
    //@ assume j != 0;
    //@ assert i==0 ==> (i%j)==0;
  }

  public void m2(int i, int j) {
    //@ assert (i<=0  && j>0) ==> (i/j) <= 0;
    //@ assert (i<=0  && j<0) ==> (i/j) >= 0;
    //@ assert (i>=0  && j>0) ==> (i/j) >= 0;
    //@ assert (i>=0  && j<0) ==> (i/j) <= 0;
    //@ assert (i==0 && j !=0) ==> (i/j)==0;
  }


}
