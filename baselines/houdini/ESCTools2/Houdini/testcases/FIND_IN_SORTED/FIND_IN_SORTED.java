public class FIND_IN_SORTED {
    
    public static int binsearch(int[] arr, int x, int start, int end) {
        if (start == end) {
            return -1;
        }
        int mid = start + (end - start) / 2; 
        if (x < arr[mid]) {
            return binsearch(arr, x, start, mid);
        } else if (x > arr[mid]) {
            return binsearch(arr, x, mid+1, end);
        } else {
            return mid;
        }
    }

    public static int find_in_sorted(int[] arr, int x) {
        return binsearch(arr, x, 0, arr.length);
    }
}
