public class ParseOld {

	int i,j;

	//@ requires i > 0;
	//@ modifies i;
	//@ ensures j == \old(i);
	//@ ensures j > 0;
	void m() {
		i = i+1;
	}
}
