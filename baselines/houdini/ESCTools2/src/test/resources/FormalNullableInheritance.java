public class FormalNullableInheritance extends FNIHelper implements FNIHelperT, FNIHelperV, FNIHelperVV {

	public void m(/*@ nullable */ Object a, /*@ nullable */ Object b, Object c); // Error on b
	public void n(/*@ nullable */ Object a, /*@ nullable */ Object b, Object c); // Error on b
	public void q(/*@ nullable */ Object d); // Error here
	public void r(/*@ nullable */ Object rd);
	public void s(/*@ nullable */ Object sd); // Error here

	public void t1(/*@ nullable */ Object tt1e);
	public void t2(/*@ nullable */ Object tt2e); // error here
	public void t3(/*@ nullable */ Object tt3e); // Error here
	public void v1(/*@ nullable */ Object tt1e);
	public void v2(/*@ nullable */ Object tt2e); // error here
	public void v3(/*@ nullable */ Object tt3e); // Error here
}

class FNIHelper extends FNIHelper2 {

	public void m(/*@ nullable */ Object a, Object b, /*@ nullable */ Object c);
	public void q(Object dd);
	public void r(/*@ nullable */ Object rdd);
	public void s(/*@ nullable */ Object sdd); // Error here

	public void v1(/*@ nullable */ Object tt1e);
	public void v2(/*@ nullable */ Object tt2e); 
	public void v3(/*@ nullable */ Object tt3e);
}

class FNIHelper2 {

	public void n(/*@ nullable */ Object a, Object b, /*@ nullable */ Object c);
	public void q(/*@ nullable */ Object qddd);
	public void r(/*@ nullable */ Object rddd);
	public void s(Object sddd);
}

interface FNIHelperT extends FNIHelperTT {

	public void t1(/*@ nullable */ Object tt1e);
	public void t2( Object tt2e);
	public void t3( Object tt3e);
}

interface FNIHelperTT {

	public void t1(/*@ nullable */ Object tt1e);
	public void t2(/*@ nullable */ Object tt2e);
	public void t3( Object tt3e);
}

interface FNIHelperV {

	public void v1(/*@ nullable */ Object tt1e);
	public void v2(/*@ nullable */ Object tt2e);
	public void v3( Object tt3e); 
}
interface FNIHelperVV {

	public void v1(/*@ nullable */ Object tt1e);
	public void v2( Object tt2e); 
	public void v3(/*@ nullable */ Object tt3e); 
}
