public class PerimeterDriver {
	/*@ requires 0 <= select && select < 6;
	  @ {|
	     @    requires select == 0;
	     @    requires 0 < shortNum && shortNum <= Short.MAX_VALUE;
	     @    ensures \result == 4*shortNum;
	     @ also 
	     @    requires select == 1;
	     @    requires 0 < w && w <= Integer.MAX_VALUE;
	     @    ensures \result == 5*w;
	     @ also 
	     @    requires select == 2;
	     @    requires 0 < longNum && 6*longNum <= Long.MAX_VALUE;
	     @    ensures \result == 6*longNum;
	     @ also 
	     @    requires select == 3;
	     @    requires 0 < w && 0 < x && 2*w + 2*x <= Integer.MAX_VALUE;
             @    ensures \result == 2*w + 2*x;
	     @ also 
	     @    requires select == 4;
	     @    requires 0 < w && 0 < x && 0 < y && w + x + y <= Integer.MAX_VALUE;
	     @    ensures \result == w + x + y;
	     @ also 
	     @    requires select == 5;
	     @    requires 0 < w && 0 < x && 0 < y && 0 < z && w + x + y + z <= Integer.MAX_VALUE;
	     @    ensures \result == w + x + y + z;
	  @ |} @*/
	 public /*@ pure @*/ long driver(int select, int w, int x, int y, int z, short shortNum, long longNum) {

		Perimeter p = new Perimeter();
		long result = 0;

		switch (select) {
		case 0:
			result = p.Perimeter(shortNum);
			break;
		case 1:
			result = p.Perimeter(w);
			break;
		case 2:
			result = p.Perimeter(longNum);
			break;
		case 3:	
			result = p.Perimeter(w, x);
			break;
		case 4:
			result = p.Perimeter(w, x, y);
			break;
		case 5:
			result = p.Perimeter(w, x, y, z);
			break;
		}
		return result;
	}
}
