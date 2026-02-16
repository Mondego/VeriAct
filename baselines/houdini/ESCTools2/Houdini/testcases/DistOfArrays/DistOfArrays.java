class DistOfArrays {
    public int distBetween(int[] arr1, int[] arr2, int d) {
        int cnt = 0;

        for (int x : arr1) {
            boolean ok = true;
            for (int y : arr2) {
                ok &= ((x>y)?(x-y):(y-x)) > d;
            }
            cnt += ok ? 1 : 0;
        }
        
        return cnt;
    }
}