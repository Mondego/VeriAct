public class Neg {
	//@ ensures \result == -num;
	public int Negation(int num) {
		//@ assume num > Integer.MIN_VALUE;
		return -num;
	}
}
