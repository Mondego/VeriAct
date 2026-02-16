// Tests bad syntax of the signals_only clause

public class SignalsOnlyBad {
        static public class A extends Exception {}
/*@
	signals_only;  // OK
	signals_only NullPointerException ,, Exception;
	signals_only , Exception;
	signals_only  String;
	signals_only Q;
*/
	public void m();


        //@ public normal_behavior
	//@ signals_only Exception;  // ERROR - not in normal_behavior
        public void mq();

        //@ signals_only A,B;  // ERROR on B - not declared
        public void mr(int i) throws A {
          if (i == 0) throw new A();
        }

        //@ signals_only A,NullPointerException;  //OK
        public void ms(int i) throws A {
          if (i == 0) throw new A();
        }
}
