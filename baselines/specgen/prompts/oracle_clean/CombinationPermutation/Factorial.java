public class Factorial
{
   public  long factorial(int n)
   {
      int c;
      long fact = 1;

   
      if (n == 0) {         
            return fact;
      }

      for (c = 1; c <= n; c++) { 
            fact = fact*c;
      }	 

      return fact;
   }

}
