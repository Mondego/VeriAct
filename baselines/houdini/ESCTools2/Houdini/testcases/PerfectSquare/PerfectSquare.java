class PerfectSquare {
    public static boolean isPerfectSquare(int num) {
        if(num == 0)
            return true;

        for(int i = 1; i * i <= num; i++) {
            if(i * i == num)
                return true;
        }
        return false;
    }
}
