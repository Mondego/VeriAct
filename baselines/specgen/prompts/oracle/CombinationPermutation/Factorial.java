public class Factorial
    {
       //@ requires 0 <= n && n <= 20;
       //@ ensures \result >= 1 && \result <= Long.MAX_VALUE;
       //@ ensures \result == spec_factorial(n);
       public /* pure @*/ long factorial(int n)
       {
          int c;
          long fact = 1;

	  //@ assert spec_factorial(0) == 1;
	   if (n == 0) {         
              return fact;
	   }

          //@ maintaining c >= 1 && c <= n+1;
	  //@ maintaining fact > 0;
	  //@ maintaining fact <= Long.MAX_VALUE; 
	  //@ maintaining spec_factorial(c - 1) == fact;
	  //@ decreases n - c;
          for (c = 1; c <= n; c++) { 
                fact = fact*c;
             }	 

          return fact;
      }

	/*@ 	requires n > 0 && n <= 20;
            	ensures 0 <= \result && \result <= Long.MAX_VALUE;
            	ensures n > 0 ==> \result == n * spec_factorial(n-1);
            also
            	requires n == 0;
            	ensures \result == 1;
        public model function static pure long spec_factorial(int n) { 
	    if (n == 0) {
		 return 1; 
	    }
	    else {
	        assert n * spec_factorial(n-1) <= Long.MAX_VALUE;
		return n * spec_factorial(n-1);
	    }
        }@*/
     }
