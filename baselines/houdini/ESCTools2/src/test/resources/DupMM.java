// Tests that duplicate bodies for model routines and classes are caught
//@ refine "DupMM.spec";

public class DupMM {

	//@ model public void m() {}

	//@ model public DupMM(int i) {}

	//@ model class C {}
}
