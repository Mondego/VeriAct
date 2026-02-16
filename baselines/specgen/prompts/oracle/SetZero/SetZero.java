public class SetZero {
	//@ requires a != null;
	//@ requires 0 <= iBegin && iBegin <= iEnd && iEnd <= a.length;
	//@ ensures (\forall int i; iBegin <= i && i < iEnd; a[i] == 0);
	public static void SetZero(int[] a, int iBegin, int iEnd) {
		int k = iBegin;
        	//@ maintaining iEnd - k >= 0;
        	//@ maintaining iBegin <= k && k <= iEnd;
        	//@ maintaining (\forall int i; iBegin <= i && i < k; a[i] == 0);
		while (k < iEnd) {
            		a[k] = 0;
            		k = k + 1 ;
        	}
	}
}
