class ConsecutiveChar {
    public int maxConsecutiveChar(String s) {
        int ans = 1, cnt = 1;

        for (int i = 1; i < s.length(); ++i) {
            if (s.charAt(i) == s.charAt(i - 1)) {
                ++cnt;
                ans = ((ans>cnt)?ans:cnt);
            } else {
                cnt = 1;
            }
        }

        return ans;
    }
}
