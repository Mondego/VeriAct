class DominantIndex {
    
    public int dominantIndex(int[] nums) {
        int m1 = -1, m2 = -1;
        int index = -1;

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > m1) {
                m2 = m1;
                m1 = nums[i];
                index = i;
            } else if (nums[i] > m2) {
                m2 = nums[i];
            }
        }

        return m1 >= m2 * 2 ? index : -1;
    }
}