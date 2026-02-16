public class StrPalindrome {
   private  String reverse = "";
   
   public boolean isPalindrome(String str) {
      
      int length = str.length();

      for (int i = length - 1; 0 <= i; i--) {
         reverse = reverse + str.charAt(i);
	 
      }
      
      return reverse.equals(str);
   }
} 
