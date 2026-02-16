// Tests bad syntax of the signals_only clause

public class SignalsOnlyBad2 {


        //@ public behavior
        //@ requires true;
        //@ {|
	//@ signals_only Exception;    // ERROR
        //@ also
        //@ signals_only NullPointerException, ClassCastException, ArrayStoreException;  
        //@ |}
        public void mmm() throws ArrayStoreException;



	//@ signals_only RuntimeException;
        //@ signals_only NullPointerException; // ERROR - not more than one
        public void mm();

	//@ signals_only RuntimeException;
        //@ signals_only NullPointerException; // ERROR - not more than one
        //@ signals_only ClassCastException; // ERROR - not more than one
        public void mm3();

}
