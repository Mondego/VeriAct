public class ModM {

	public int i,j,k;

	//@ requires i >=0;
	//@ modifies i,j;
	//@ also
	//@ requires i <= 0;
	//@ modifies i,k;
	public void m() {
		mm();       // WARNING
	}


	//@ requires i >=0;
	//@ modifies j;
	//@ also
	//@ requires i <= 0;
	//@ modifies i,k;
	public void mm();

	//@ modifies \nothing;
	public void me() { mee(); } // ERROR

	//@ modifies \everything;
	public void mee();

	//@ requires i >=0;
	//@ modifies i,j;
	//@ also
	//@ requires i <= 0;
	//@ modifies i,k;
	public void mp() {
		if (i > 0) mmp(); // OK
		else if (i < 0) mmp(); // OK
	}

	//@ requires i >=0;
	//@ modifies i,j;
	//@ also
	//@ requires i <= 0;
	//@ modifies i,k;
	public void mp2() {
		if (i == 0) mmp(); // WARNING
	}
	//@ requires i >=0;
	//@ modifies j;
	//@ also
	//@ requires i <= 0;
	//@ modifies i,k;
	public void mmp();

	//@ requires i == 1;
	//@ modifies i,j;
	public void mq() {
		i = 2;
		mqq(); // OK
	}

	//@ requires i == 2;
	//@ modifies j;
	//@ also
	//@ requires i == 1;
	//@ modifies \nothing;
	public void mqq();
}


