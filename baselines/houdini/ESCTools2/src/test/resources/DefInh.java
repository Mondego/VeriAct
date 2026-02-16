// This tests that default constructors have the same modifies clause as
// their parent constructors.  Watch that variable references are properly
// resolved even when there are hiding declarations.

public class DefInh extends DefS {

 int si;

}

class DefS {

	// modifies ((DefS)this).si,sj;
	//@ modifies si,sj;
	public DefS() {}

	int si;
	static int sj;

	//@ modifies si;
	public void m(){
		si = 0;
	}
}

class DefD extends DefInh {

	//@  modifies sj, ((DefS)this).si;
	public DefD() {  // FIXME - should be ok
	}


	//@ modifies ((DefS)this).si;
	public void m() {
		super.m();
	}

}

class DefIS extends DefSS {
}

class DefII extends DefIS {
}

class DefDD extends DefII {

	//@ modifies si;
	public DefDD() {}
}

class DefSS {

	int si;
	//@ modifies si;
	public DefSS() { si = 0; }
}

