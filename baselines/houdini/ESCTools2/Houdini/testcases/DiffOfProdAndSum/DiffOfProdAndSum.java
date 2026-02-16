class DiffOfProdAndSum {
    public int diffOfProdAndSum(int n) {
        int multiply = 1, add = 0;

        while (n > 0) {
            multiply *= n % 10;
            add += n % 10;
            n /= 10;
        }
        
        return multiply - add;
    }
}