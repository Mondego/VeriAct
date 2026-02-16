class CommonFactorsNum {
    public int commonFactorsNum(int a, int b) {
        int ans = 0;

        for (int x = 1; x <= a && x <= b; ++x) {
            if (a % x == 0 && b % x == 0) {
                ++ans;
            }
        }
        
        return ans;
    }
}