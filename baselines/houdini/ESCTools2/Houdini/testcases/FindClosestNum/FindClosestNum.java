import java.util.Math;

class FindClosestNum {
    
    public int findClosestNumber(int[] nums) {
        int ans = Integer.MAX_VALUE;

        for(int num : nums) {
            int absNum = Math.abs(num);
            if(absNum < Math.abs(ans) || (absNum == Math.abs(ans) && num > ans)) {
                ans = num;
            }
        }

        return ans;
    }
}