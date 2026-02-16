class PrimeCheck {
   /*@ public normal_behavior     
     @    requires d != 0;
     @    ensures \result == n%d; @*/
   public /*@ pure @*/ static int div(int n, int d) { return n%d; }

   //@ requires 1 < a;
   //@ ensures \result ==> (\forall int k; 1 < k && k <= a/2; div(a, k) != 0);
   //@ ensures !\result ==> (\exists int k; 1 < k && k <= a/2; div(a,k) == 0); 
   public boolean isPrime(int a) {
	
	int i = 2;
	int mid = a/2;

	//@ ghost int maxnumber = Integer.MAX_VALUE;	
	//@ maintaining 1 < i && i <= mid + 1;
	//@ maintaining 2 < i ==> \forall int k; 1 < k && k < i; div(a, k) != 0; 
	//@ decreases maxnumber - i;
	while (i <= mid) {
	   if (div(a,i) == 0)
		return false;
	   i++;
	}
	return true;
   }
}
