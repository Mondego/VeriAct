public class AbsoluteDriver {
	 int i;
	 short sh;
	 long l;

	public  AbsoluteDriver(short sh, int i, long l) {
		this.sh = sh;
		this.i = i;
		this.l = l;
	}
	
	public void driver() {
		Absolute p = new Absolute();
		this.sh = p.Absolute(sh);
		this.i = p.Absolute(i);
		this.l = p.Absolute(l);
	}
}
