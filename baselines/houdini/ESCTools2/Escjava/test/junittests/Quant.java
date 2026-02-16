// Tests various quantifiers, checking that scopes are ok

public class Quant {

	//@ invariant (\forall int i;; 0<i && i<4 && i==2);
	//@ invariant (\exists int i;; 0<i && i<4 && i==2);
	//@ invariant (\forall int i; 0<i && i<4 && i==2);
	//@ invariant (\exists int i; 0<i && i<4 && i==2);
	//@ invariant (\exists int i, j; 0<i && i< j && j<4  && i==j);
	//@ invariant (\forall int i; 0<i && i<4 ; i==2);
	//@ invariant (\exists int i; 0<i && i<4 ; i==2);
	//@ invariant (\exists int i, j; 0<i && i< j && j<4 ; i==j);

	//@ invariant (\max int i; 0<i && i<4; i) == 3;
	//@ invariant (\min Quant s; s.n()<10; s.n() ) == 1;
	//@ invariant (\min int i; 0<i && i<4; i) == 1;
	//@ invariant (\sum int i; 0<i && i<4; i) == 6;
	//@ invariant (\product int i; 0<i && i<4; i) == 6;
	//@ invariant 1 == (\num_of int i; 0<i && i<4; i==2);
	//@ invariant 1 == (\num_of int i; 0<i && i<4  && i==2);

	//@ invariant (\sum int i,j; 0<i && i<j && j<4; i+j) == 6;
	//@ invariant 1 == (\num_of int i,j; 0<i && i<j && j<4 && i==j);

	//@ invariant (\min int i; i) == 1;
	//@ invariant (\min int i;; i) == 1;

	public void m() {}

	//@ pure
	public int n() { return 0; }
}


