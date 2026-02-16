
// FIXME - the failure in mconstr appears to be a failure in the prover.

public class StringFresh {

StringFresh ();

//@ ensures (st + stt) !=null;
//@ ensures (st + stt) != st;
//@ ensures (st + stt) != (st + stt);
//@ pure
public void mp(/*@ non_null */ String st, String stt) {
}

public void m(/*@ non_null */ String st, String stt) {
	String s,ss,sss;
	Object o;
	ss = s + o;
	sss = s + o;
	//@ assert ss == ss;    // OK
	//@ assert ss != sss;   // OK

	ss.hashCode();  // OK

}
//@ pure
public void mm(/*@ non_null */ String st) {
	st.hashCode();  // FAILS
}

public String f = new String();

public void meq(){
	String s = "";
	String ss = s;
	ss += s;
	//@ assert ss != s;
	//@ assert ss == ss;
	f += s;
	//@ assert f != s;
}

public String ff = "";

public void mconstr(){
	String s = new String();
	ff += s;
}

public void mlit() {
	String s = "abcdefghijklmnopqrstuvwxyz";
	String s2 = "abcdefghijklmnopqrstuvwxyz";
	String s3 = "abcdefghijklmnopqrstuvwxzy";
	String ss = s + s;
	//@ assert s == s2;
	//@ assert s != s3;
	//@ assert ss != s;

}
}
