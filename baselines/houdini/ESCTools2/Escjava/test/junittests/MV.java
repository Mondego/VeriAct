/* This tests that modified model vars without representations
   work like modified ghost vars. */

public class MV {
     MV();

    //@ public static model int t;
    //@ public static ghost int tt;

    //@ public normal_behavior
    //@   modifies t;
    //@   ensures t >= \old(t);
    //@   ensures \result == t;
    static public int value();

    //@ public normal_behavior
    //@   modifies tt;
    //@   ensures tt >= \old(tt);
    //@   ensures \result == tt;
    static public int gvalue();
}

class Client {
    Client();

   public void m() {
	int i = MV.gvalue();
        int ii = MV.gvalue();
        //@ assert ii >= i;
   }

   public void mm() {
	int i = MV.value();
        int ii = MV.value();
        //@ assert ii >= i;
   }
}
