// Tests the loop invariant
//#FLAGS: -loop 3

public class LoopInvariant {

	public void m() {
		int sum = 0;
		//@ loop_invariant sum == i;
		//@ decreases -i;
		for (int i=0; i<10; ++i) sum++;
	}

	public void m2() {
		int sum = 0;
		//@ loop_invariant sum == i;
		//@ decreases i;
		for (int i=0; i<10; ++i) sum++;
	}

	public void mm() {
		int sum = 0;
		int i = 0;
		//@ loop_invariant sum == i;
		while (i<5) {
			i++; 
		}
	}

	public void mmm() {
		int sum = 0;
		//@ loop_invariant sum == 0;
		//@ decreases 10-i;
		for (int i=0; i<10; ++i) sum++;
	}
}
