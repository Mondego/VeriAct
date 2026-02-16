public class Fibonacci {
   	private  long fib[];

	Fibonacci() {
		fib = new long[2];
		fib[0] = 0;
		fib[1] = 1;
	}
	 	
	 Fibonacci(int size) {
		if (2 <= size && size <= 93) {
			fib = new long[size];	
			fib[0] = 0;
			fib[1] = 1;
		} else {
			throw new IllegalArgumentException();
		}
	}

	public  long getFib(int index) {
		return fib[index];
	}

	public void fibCompute() {
		int index = 2;
      
		while (index < fib.length) {
	
			fib[index] = fib[index - 2] + fib[index - 1];
			index++;
                        
		}
	}
}
