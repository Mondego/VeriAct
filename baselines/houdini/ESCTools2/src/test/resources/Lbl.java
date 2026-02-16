// Tests the lblneg and lblpos functions

public class Lbl {

	//@ ensures (\lblneg A false);
	void m() {}

	void mm() {
	//@ assert !(\lblpos B true);
	}
}
