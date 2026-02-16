public class Absolute {
	
	public  short Absolute(short num) {
		if (0 <= num)
			return num;
		else
			return (short)-num;	
	}

	
	public  int Absolute(int num) {
		if (0 <= num)
			return num;
		else
			return -num;
	}

	
	public  long Absolute(long num) {
		if (0 <= num)
			return num;
		else
			return -num;	
	}
}
