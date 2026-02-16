//#FLAGS: -testMode -showFields -quiet -classpath . -specs ../../../specs
public class Scope extends ScopeS implements ScopeI {
//@	model int k; // ERROR - duplicates a java field

	public int i; // ERROR - duplicates a jml ghost field
	public int j = jj; // Hides an interface field ScopeII.j, inherited by two paths; jj is ok as well
	public static int sj = jj; // jj should be static

	public int k; // ERROR - duplicates a jml model field
/*@
	ghost public int gi; // Does not hide anything
	ghost public int i; // ERROR - duplicates a java field
	model public int gsi; // hides java field ScopeS.gsi
	model public int ggi; // hides jml field ScopeS.ggi
*/
	public int si = gsi; // should see the Java field ScopeS.gsi

	//@ ghost public int z = dup; // ERROR - dup is in two jml parents
	public int zz = dupj; // ERROR - dupj is in two java parents
	public int zzz = dupm; // OK - only sees the java parent
	//@ ghost public int zzzz = dupm; // ERROR - is in a java and a jml parent
	//@ ghost public int z5 = dupmm; // ERROR - FIXME
}
interface ScopeI extends ScopeII {

	static public int dupj; 
	//@ ghost public int dup;
	//@ ghost public int dupm;
	public int dupmm;
}

interface ScopeII {
	final int j = 0;
	final int jj = 0;

}

class ScopeS implements ScopeII {

	public int i;
	public int gsi;
	static public int dupj;
	static public int dupm;
	private int priv;
//@	ghost public int ggi;
//@	ghost public int dup;
//@	ghost private int gpriv;
//@	ghost public int dupmm;
}
