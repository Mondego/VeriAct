public class GhostFieldOverride extends CHA {

	//@ ghost public int s; // OK
	//@ ghost public int i; // OK

	int z;
	//@ ghost public int z;
	int y;
	//@ model public int y;
	//@ ghost int x;
	//@ model int x;
}

class CHA {

	static public int i;
	//@ ghost static public int s;
}
