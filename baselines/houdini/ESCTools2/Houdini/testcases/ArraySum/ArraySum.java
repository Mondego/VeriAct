class ArraySum {
    public static int arraySum(int[] arr) {
        int res = 0;

        for(int i = 0; i < arr.length; i++) {
            res += arr[i];
        }
        
        return res;
    }
}