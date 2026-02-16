public class GCD {
	
        public static int div(int n, int d) { 
		return n%d; 
	}

	public  int absolute(int num) {
		return (0 <= num) ? num : -num;
	}
    
	public  int gcd(int num1, int num2) throws IllegalArgumentException {
		int result = 1; 
	 	num1 = absolute(num1);
		num2 = absolute(num2);

		if (num1 == 0 && num2 == 0) {
			return -1;	
		}

		if (num1 == 0 || num2 == 0) { 
			return (num1 > num2) ? num1 : num2;
		}

		for (int i = 1; i <= num1 && i <= num2; i++) {
            		if (div(num1,i) == 0 && div(num2,i) == 0) {
               			result = i;
			}
        }
		return result;
	} 
}
