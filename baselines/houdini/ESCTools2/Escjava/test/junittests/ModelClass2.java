//#FLAGS: -classpath . -source 1.4 
public class ModelClass2 {

/*@
    public model class M {
	public int i;
	public non_null Object o = new Object();
	invariant i > 0;  nowarn ;
	protected invariant i > 0;

	requires i > 1;
	also
	ensures true;
	protected void m();


	{|
		requires true;
	also
		ensures true;
	|}
	pure
	public void n();

	ghost public int jj,j,jjj;
	public model float ff,f,fff;
	public invariant j == 0;

	requires true;
	public void p(int i) {
		set j = 0; //@ nowarn Modifies;
		int k = i+1;
		assert k == 1;
		assume k == 1;
		ghost int kk = 0;
	}

	public M() { i = 1; } //@ nowarn Modifies;
    }
*/

	//@ ghost int j = 0;

	void mjava() {}

	/*@ model void z(int i) {

		set j = 0; //@ nowarn Modifies;
		int k = i+1;
		assert k == 1;
		assume k == 1;
		ghost int kk = 0;

	}
	*/
}

/* FIXME
- nowarn on line 7 seems to turn off warnings on line 8 as well
- nothing with the POST modifier pragmas (in, maps)
- what about method, field, constructor keywords
- what about ghost decls in which ghost is not first
- what about ghost decls in which there are modifiers and pmodifiers (e.g. non_null)
*/
