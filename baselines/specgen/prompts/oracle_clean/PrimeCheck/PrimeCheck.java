class PrimeCheck {
   
   public static int div(int n, int d) { return n%d; }

   public boolean isPrime(int a) {
	
	int i = 2;
	int mid = a/2;

	while (i <= mid) {
	   if (div(a,i) == 0)
		return false;
	   i++;
	}
	return true;
   }
}
