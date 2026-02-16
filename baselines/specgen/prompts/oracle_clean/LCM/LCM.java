public class LCM {
    
    public static int div(int n, int d) { return n%d; }

    public  int lcm(int num1, int num2) 
    {
	if (num1 == 0 || num2 == 0) {
		return -1;
	}	
	if (num1 < 0)
		num1 = -num1;
	if (num2 < 0)
		num2 = -num2;

        int result = (num1 > num2) ? num1 : num2;

        while (result < Integer.MAX_VALUE)
        {
            if (div(result, num1) == 0 && div(result, num2) == 0)
            {
                break;
            }
            result++;
        }

	if (div(result, num1) == 0 && div(result, num2) == 0) {
		return result;
	}
	return -1;
    }
}
