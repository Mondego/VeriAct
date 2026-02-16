public class FormalNonnullInheritance extends FMIHelper implements FMIHelperT, FMIHelperV, FMIHelperVV {

	public void m(/*@ non_null */ Object a, /*@ non_null */ Object b, Object c); // Error on b
	public void n(/*@ non_null */ Object a, /*@ non_null */ Object b, Object c); // Error on b
	public void q(/*@ non_null */ Object d); // Error here
	public void r(/*@ non_null */ Object rd);
	public void s(/*@ non_null */ Object sd); // Error here

	public void t1(/*@ non_null */ Object tt1e);
	public void t2(/*@ non_null */ Object tt2e); // error here
	public void t3(/*@ non_null */ Object tt3e); // Error here
	public void v1(/*@ non_null */ Object tt1e);
	public void v2(/*@ non_null */ Object tt2e); // error here
	public void v3(/*@ non_null */ Object tt3e); // Error here
}

class FMIHelper extends FMIHelper2 {

	public void m(/*@ non_null */ Object a, Object b, /*@ non_null */ Object c);
	public void q(Object dd);
	public void r(/*@ non_null */ Object rdd);
	public void s(/*@ non_null */ Object sdd); // Error here

	public void v1(/*@ non_null */ Object tt1e);
	public void v2(/*@ non_null */ Object tt2e); 
	public void v3(/*@ non_null */ Object tt3e);
}

class FMIHelper2 {

	public void n(/*@ non_null */ Object a, Object b, /*@ non_null */ Object c);
	public void q(/*@ non_null */ Object qddd);
	public void r(/*@ non_null */ Object rddd);
	public void s(Object sddd);
}

interface FMIHelperT extends FMIHelperTT {

	public void t1(/*@ non_null */ Object tt1e);
	public void t2( Object tt2e);
	public void t3( Object tt3e);
}

interface FMIHelperTT {

	public void t1(/*@ non_null */ Object tt1e);
	public void t2(/*@ non_null */ Object tt2e);
	public void t3( Object tt3e);
}

interface FMIHelperV {

	public void v1(/*@ non_null */ Object tt1e);
	public void v2(/*@ non_null */ Object tt2e);
	public void v3( Object tt3e); 
}
interface FMIHelperVV {

	public void v1(/*@ non_null */ Object tt1e);
	public void v2( Object tt2e); 
	public void v3(/*@ non_null */ Object tt3e); 
}
