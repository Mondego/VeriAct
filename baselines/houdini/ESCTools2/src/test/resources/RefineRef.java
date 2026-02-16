
public class RefineRef {
	//@ requires r != null;
	public void m(RefineRefHelper r) {
		r.o.get();
	}
	//@ requires r != null;
	public void mm(RefineRefHelper r) {
		r.oo.get();
	}
	//@ requires r != null;
	public void mmm(RefineRefHelper r) {
		r.ooo.get();
	}
}
