class MyPower {
    public static double power(double x, int n) {
        double res = 1.0;
        for(int i = 0; i < n; i++)
	    res = res * x;
        return res;
    }
}
