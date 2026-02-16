public class SignalsBad {




    //@ signals (NullPointerException) true; // ERROR - DOES NOT MATCH
    public void m() throws java.io.IOException;

    //@ signals_only RuntimeException;
    //@ signals (NullPointerException) true; // OK
    public void n() throws java.io.IOException;

    //@ signals_only NullPointerException;
    //@ signals (NullPointerException) true;  // OK
    public void p();

    //@ signals_only java.io.IOException;
    //@ signals (NullPointerException) true;  // ERROR
    public void r() throws Exception;

    //@ signals_only StringIndexOutOfBoundsException;
    //@ signals (NullPointerException) true;  // ERROR
    public void s();

    //@ signals (NullPointerException) true;  // ERROR - no signals_only or throws
    public void q();

    //@ signals_only \nothing;
    public void t();

    //@ signals_only ;   // CAUTION
    public void z();




}
