package sub;

public class JavaScopeBB {
	class Inn {
		//@ requires System.out != null;
		//@ ensures \result == 1;
		public int m() { System.out.println("A"); return 1; }

	}
}
