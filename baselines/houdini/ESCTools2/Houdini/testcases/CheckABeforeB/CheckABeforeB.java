class CheckABeforeB {
    public boolean checkString(String s) {
        char[] chars = s.toCharArray();
        int i = 0;
        while (i < chars.length && chars[i] == 'a') {
            i++;
        }
        while (i < chars.length && chars[i] == 'b') {
            i++;
        }
        return i == chars.length;
    }
}