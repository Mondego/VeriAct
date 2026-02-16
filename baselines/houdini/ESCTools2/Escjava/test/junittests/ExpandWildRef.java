// This tests that wild references are expanded into the corresponding fields
// when used in a called routine.
// FIXME - add model fields ???
public class ExpandWildRef extends EWRS implements EWRI {

	//@ non_null
	Other o;

	ExpandWildRef();

	public int ii;
	public static int sii;
	//@ public ghost int gii;
	//@ public static ghost int sgii;

	//@ modifies this.*;
	//@ ensures o == \old(o);
	void mm();

	//@ modifies ExpandWildRef.*;
	void mms();

	//@ modifies Other.*;
	void mos();

	//@ modifies o.*;
	void mo();

	void p() {
		ii = 1;
		sii = 1;
		//@ set gii = 1;
		//@ set sgii = 1;
		ewr = 1;
		ewrs = 1;
		//@ set iewri = 1;
		//@ set sewri = 1;
		ewrj = 1;
		o.i = 1;
		o.si = 1;
		//@ set o.gi = 1;
		//@ set o.sgi = 1;
		mm();
		//@ assert ii == 1 || gii == 1 || ewr == 1 || iewri == 1;//ERROR
	}		
	void p2() {
		ii = 1;
		sii = 1;
		//@ set gii = 1;
		//@ set sgii = 1;
		ewr = 1;
		ewrs = 1;
		//@ set iewri = 1;
		//@ set sewri = 1;
		ewrj = 1;
		o.i = 1;
		o.si = 1;
		//@ set o.gi = 1;
		//@ set o.sgi = 1;
		mm();
		//@ assert sii == 1;
		//@ assert sgii == 1;
		//@ assert ewrs == 1;
		//@ assert sewri == 1;
		//@ assert o.i == 1;
		//@ assert o.si == 1;
		//@ assert o.gi == 1;
		//@ assert o.sgi == 1;
		//@ assert ewrj == 1;
	}		
	void pp() {
		ii = 1;
		sii = 1;
		//@ set gii = 1;
		//@ set sgii = 1;
		ewr = 1;
		ewrs = 1;
		//@ set iewri = 1;
		//@ set sewri = 1;
		ewrj = 1;
		o.i = 1;
		o.si = 1;
		//@ set o.gi = 1;
		//@ set o.sgi = 1;
		mms();
		//@ assert sii == 1 || sgii == 1 || ewrs == 1 || sewri == 1 || ewrj == 1; // ERROR
	}		
	void pp2() {
		ii = 1;
		sii = 1;
		//@ set gii = 1;
		//@ set sgii = 1;
		ewr = 1;
		ewrs = 1;
		//@ set iewri = 1;
		//@ set sewri = 1;
		ewrj = 1;
		o.i = 1;
		o.si = 1;
		//@ set o.gi = 1;
		//@ set o.sgi = 1;
		mms();
		//@ assert ii == 1 && gii == 1 && ewr == 1 && iewri == 1 && o.i == 1 && o.si == 1 && o.gi == 1 && o.sgi == 1;
	}		
	void po() {
		ii = 1;
		sii = 1;
		//@ set gii = 1;
		//@ set sgii = 1;
		ewr = 1;
		ewrs = 1;
		//@ set iewri = 1;
		//@ set sewri = 1;
		ewrj = 1;
		o.i = 1;
		o.si = 1;
		//@ set o.gi = 1;
		//@ set o.sgi = 1;
		mo();
		//@ assert o.i == 1 || o.gi == 1 ; // ERROR
	}		
	void po2() {
		ii = 1;
		sii = 1;
		//@ set gii = 1;
		//@ set sgii = 1;
		ewr = 1;
		ewrs = 1;
		//@ set iewri = 1;
		//@ set sewri = 1;
		ewrj = 1;
		o.i = 1;
		o.si = 1;
		//@ set o.gi = 1;
		//@ set o.sgi = 1;
		mo();
		//@ assert o.si == 1 && o.sgi == 1 ;
		//@ assert sii == 1 && sgii == 1 && ewrs == 1 && sewri == 1;
		//@ assert ii == 1 && gii == 1 && ewr == 1 && iewri == 1;
		//@ assert ewrj == 1;
	}
	void pos() {
		ii = 1;
		sii = 1;
		//@ set gii = 1;
		//@ set sgii = 1;
		ewr = 1;
		ewrs = 1;
		//@ set iewri = 1;
		//@ set sewri = 1;
		ewrj = 1;
		o.i = 1;
		o.si = 1;
		//@ set o.gi = 1;
		//@ set o.sgi = 1;
		mos();
		//@ assert o.si == 1 || o.sgi == 1;  // ERROR
	}		
	void pos2() {
		ii = 1;
		sii = 1;
		//@ set gii = 1;
		//@ set sgii = 1;
		ewr = 1;
		ewrs = 1;
		//@ set iewri = 1;
		//@ set sewri = 1;
		ewrj = 1;
		o.i = 1;
		o.si = 1;
		//@ set o.gi = 1;
		//@ set o.sgi = 1;
		mos();
		//@ assert o.i == 1 && o.gi == 1;
		//@ assert sii == 1 && sgii == 1 && ewrs == 1 && sewri == 1;
		//@ assert ii == 1 && gii == 1 && ewr == 1 && iewri == 1;
		//@ assert ewrj == 1;
	}		
}

class EWRS {

	public int ewr;
	static public int ewrs;

}

interface EWRI {

	public int ewrj;
	//@ ghost public int sewri; // static by default
	//@ instance ghost public int iewri;

}

class Other {

	static public int si;
	public int i;
	//@ ghost public int gi;
	//@ ghost static public int sgi;

}
