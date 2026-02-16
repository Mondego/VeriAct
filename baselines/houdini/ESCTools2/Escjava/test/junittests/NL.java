//#FLAGS: 
// Tests the behavior of non_null under inheritance
public class NL extends NLS {
	Object oo; // CASE 2:
	// Parent has non_null
	// No spec, so inherit the full spec of the parent method
	public void m(Object o) {
		o = null;  // OK
	}
	// Parent has non_null
	// No spec, so inherit the full spec of the parent method
	public void m2(Object o) {
		nonnull(o); // OK
	}
	public void mm1() {
		m(oo);  // FAILS
	}
	public void mm2() {
		(new NLS()).m(oo);  // FAILS
	}
// CASE 1:
	// Parent has non_null
	//@ also
	//@ requires true;
	public void qm2(Object o) {
		nonnull(o); // FAILS 
	}
	// Parent has non_null
	//@ also
	//@ requires true;
	/*@ pure */ public void qm(Object o) {
		o = null;  // OK
	}
	public void qmm1() {
		qm(oo); // OK 
	}
	public void qmm2() {
		(new NLS()).qm(oo);   // FAILS
	}
// CASE 1b:
	// Parent has non_null
	//@ also
	//@ requires false;
	public void qqm2(Object o) {
		nonnull(o); // OK 
	}
	// Parent has non_null on the formal argument
	//@ also
	//@ requires false;
	/*@ pure */ public void qqm(Object o) {
		o = null;  // OK
	}
	public void qqmm1() {
		qqm(oo); // FAILS 
	}
	public void qqmm2() {
		(new NLS()).qqm(oo);   // FAILS
	}
// CASE 3:
	// Parent is not non_null
	public void n2(/*@ non_null */ Object o) {
		nonnull(o); // FAILS since non_null is ignored
	}
	// Parent is not non_null
	/*@ pure */ public void n(/*@ non_null */ Object o) {
		o = null;  // OK since non_null is ignored
	}
	public void nn() {
		n(oo);  // OK
	}
	public void nn2() {
		(new NLS()).n(oo);  // OK
	}

	/*@ pure */ public void p(/*@ non_null */ Object o) {
		o = null;  // FAILS
	}
	public void p2(/*@ non_null */ Object o) {
		nonnull(o); // OK
	}
	public void pp() {
		p(oo);  // FAILS
	}
	public void pp2() {
		(new NLS()).p(oo); // FAILS 
	}

}

class NLS {
	/*@ pure */ public void nonnull(/*@ non_null */ Object o);

	/*@ pure */ public void m(/*@ non_null */ Object o) {
		o = null;  // FAILS
	}
	public void m2(/*@ non_null */ Object o) {
	}
	/*@ pure */ public void qm(/*@ non_null */ Object o) {
		o = null;  // FAILS
	}
	public void qm2(/*@ non_null */ Object o) {
	}
	/*@ pure */ public void qqm(/*@ non_null */ Object o) {
		o = null;  // FAILS
	}
	public void qqm2(/*@ non_null */ Object o) {
	}
	/*@ pure */ public void n(Object o) {
		o = null;  // OK
	}
	public void n2(Object o) {
		nonnull(o); // FAILS
	}
	/*@ pure */ public void p(/*@ non_null */ Object o) {
		o = null;  // FAILS
	}
	public void p2(/*@ non_null */ Object o) {
		nonnull(o); // OK
	}

	//@ pure
	public NLS();
}
