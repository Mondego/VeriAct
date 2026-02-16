public class CombinationPermutation {

    private  long combination(int n, int r) {
		Factorial fac = new Factorial();
		long combin;
		combin = fac.factorial(n) / (fac.factorial(r) * fac.factorial(n-r));
		return combin;
	}

	private  long permutation(int n, int r) {
		Factorial fac = new Factorial();
		long permut;
		permut = fac.factorial(n) / fac.factorial(n-r);
		return permut;
	}

	public  long select(int n, int r, boolean flag) {
		return flag ? combination(n, r) : permutation(n, r);
	}
}

