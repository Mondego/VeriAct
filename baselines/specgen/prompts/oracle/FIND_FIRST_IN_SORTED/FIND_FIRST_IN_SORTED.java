//arr.length should be less than half of Integer.MAX_VALUE. Array size greater than ((Integer.MAX_VALUE/2)+1) can lead to sum overfllowing  for "int mid = (lo + hi) / 2;". When argument "X" is not in the array and it is value is larger than all numbers in the array. 
public class FIND_FIRST_IN_SORTED {
    //@ requires 0 <= arr.length && arr.length <= (Integer.MAX_VALUE/2)+1; 
    /*@ requires (\forall int j; 0 <= j && j < arr.length;
      @             (\forall int i; 0 <= i && i < j ; arr[i] <= arr[j])); @*/
    //@ ensures \result < arr.length;
    //@ ensures 0 <= \result && \result < arr.length ==> arr[\result] == x && (\forall int i; 0 <= i && i < \result; arr[i] != x);
    //@ ensures \result == -1 ==> (\forall int i; 0 <= i && i < arr.length; arr[i] != x);
    public static int find_first_in_sorted(int[] arr, int x) {
        int lo = 0;
        int hi = arr.length;

	//@ maintaining 0 <= lo && lo <= arr.length; 
        //@ maintaining 0 <= hi && hi <= arr.length;
	//@ maintaining lo <= hi;
        //@ maintaining (\forall int i; 0 <= i && i < lo; arr[i] < x);
        //@ maintaining (\forall int i; hi < i && i < arr.length; x <= arr[i]);
        while (lo < hi) {
            int mid = (lo + hi) / 2; // check if this is floor division
            if (x == arr[mid] && (mid == 0 || x != arr[mid-1])) {
                return mid;
            } else if (x <= arr[mid]) {
                hi = mid;
            } else { 
                lo = mid + 1;
            }
        }
	//@ assume (\forall int i; 0 <= i && i < arr.length; arr[i] != x);
        return -1;
    }
}
