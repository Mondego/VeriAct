// This test has a repeated use of variable e .  Javac and jml are ok with 
// this, but escjava complains.

public class EX {

	public int mmm(int i) {

	    try {
		if (i==0) throw new Exception();
	    } catch (Exception e) {

		Object o = new Object() {
		    public int hashcode() {
		      try {
			int k = 2;
			if (k==0) throw new Exception();
			return 0;
		      } catch (Exception e) {
			return 0;
		      }
		    }
		};
	    }
	    return 0;
	}
}
