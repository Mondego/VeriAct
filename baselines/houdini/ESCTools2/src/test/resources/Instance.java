//#FLAGS: -nocheck
public class Instance implements I {
	//@ invariant b && ib && mb && mib && m() && im();
	//@ instance invariant b && ib && mb && mib && m() && im();
}

interface I {

	//@ ghost boolean b;
	//@ instance ghost boolean ib;

	//@ model boolean mb;
	//@ instance model boolean mib;

	//@ model boolean m();
	//@ model boolean im();

	//@ invariant b && ib;
	//@ instance invariant ib && b;
}
