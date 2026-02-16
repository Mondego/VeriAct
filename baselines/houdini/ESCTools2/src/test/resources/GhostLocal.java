public class GhostLocal {
	public GhostLocal();

	//@ ghost public int i;
	public int ii = 1;

	//@ requires ii == 1;
	//@ requires i == 0;  //@ modifies i;
	public void m() {

		//@ set i = 1;		
		int j = 10;
		//@ ghost int k = i+j;
		//@ set i = 3;	
		//@ assert k == 11;
		//@ assert k == i+j-2;
		//@ ghost int kk = ii;
		//@ set k = k + 1;
		//@ assert k == 12;
	}

	//-@ function
	public boolean pred(int i);

	public void mm() {
		//@ ghost int i = (\max int ii; 0<ii && ii<4; ii+2);
		//@ ghost int j = (\num_of int jj; 0<jj && jj < 4);
		// ghost boolean b = (\exists int bb;  pred(bb));
		//@ set i = (\max int ii; 0<ii && ii<4; ii+2);
		//@ set j = (\num_of int jj; 0<jj && jj < 4);
		int z;
	}
}

class GhostLocalA extends GhostLocalS implements GhostLocalI {
	public GhostLocalA a = new GhostLocalA();
	public GhostLocalA b = new GhostLocalA();
	//@ modifies a.i, a.j, b.i, b.j;
	//@ requires a != null && b != null && a != b;
	public void m() {
		//@ set a.i = 1;
		//@ set a.j = 2;
		//@ set b.i = 3;
		//@ set b.j = 4;
		//@ assert a.i == 1; // OK if no aliasing!
		//@ assert a.j == 4;
		//@ assert a.i == 3; // ERROR
	}
}

class GhostLocalS implements GhostLocalI {
}

interface GhostLocalI {
	//@ ghost instance int i;
	//@ ghost static int j;
}
