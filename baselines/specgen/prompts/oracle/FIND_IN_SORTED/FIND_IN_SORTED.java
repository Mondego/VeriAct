public class FIND_IN_SORTED {
    /*@ requires  0 <= start && 0 <= end && start <= end && end <= arr.length; 
      @ requires (\forall int j; 0 <= j && j < arr.length; 
      @                        (\forall int i; 0 <= i && i < j; arr[i] <= arr[j]));
      @ ensures (0 <= \result && \result < arr.length) ==> arr[\result] == x;
      @ ensures (start < end && 0 <= \result && \result < end) 
      @     ==> (arr[start] <= arr[\result] && arr[\result] <= arr[end-1]);
      @ ensures \result < end;
      @ ensures (start == end) ==> \result == -1;
      @ ensures \result == -1 ==> (\forall int i; start <= i && i < end; arr[i] != x); @*/
    public static /*@ pure @*/
    int binsearch(int[] arr, int x, int start, int end) {
        if (start == end) {
            return -1;
        }
        int mid = start + (end - start) / 2; // check this is floor division
        if (x < arr[mid]) {
            return binsearch(arr, x, start, mid);
        } else if (x > arr[mid]) {
            return binsearch(arr, x, mid+1, end);
        } else {
            return mid;
        }
    }

    //@ requires \forall int j; 0 <= j && j < arr.length; \forall int i; 0 <= i && i < j ;arr[i] <= arr[j];
    //@ ensures 0 <= \result && \result < arr.length ==> arr[\result] == x;
    //@ ensures \result == -1 ==> (\forall int i; 0 <= i && i < arr.length; arr[i] != x);
    public static int find_in_sorted(int[] arr, int x) {
        return binsearch(arr, x, 0, arr.length);
    }
}
