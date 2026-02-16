class NegCounter {

    public countNegInArray(int[] arr) {
        int cnt = 0;

        for (int i = 0; i < arr.length; i++) {
            if(arr[i] < 0)
                cnt = cnt + 1;
        }

        return cnt;
    }

}