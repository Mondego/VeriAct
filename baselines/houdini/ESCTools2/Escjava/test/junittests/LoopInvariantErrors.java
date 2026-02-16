

public class LoopInvariantErrors {

	void m() {
		int i;
		//@ loop_invariant i;
		//@ decreases i > 0;
		for (int j=0; j<10; ++j) {}
	}

	void mm() {
		//@ loop_invariant true;
		int i;
		//@ decreases i;
		++i;
	}
}
