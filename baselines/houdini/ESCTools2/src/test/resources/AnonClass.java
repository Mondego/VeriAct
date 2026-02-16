public class AnonClass {

    public void p() {
	AnonClassS a = new AnonClassS() {
		//@ also
		//@ ensures \result == 2;
		public int m(int i) { return 3; } // FIXME - not checked
        };

	//@ assert \typeof(a) != \type(AnonClassS);
	a.m(-1);

    }
}

class AnonClassS {

    //@ requires i > 0;
    public int m(int i) { return 1; }
}
