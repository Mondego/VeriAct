// Tests spec_public, spec_protected
public class Privacy {
	private boolean b;

	//@ spec_public
	private boolean bb;

	//@ spec_protected
	private boolean bbb;

	boolean pb;

	//@ spec_public
	boolean pbb;

	//@ spec_protected
	boolean pbbb;

	protected boolean ppb;

	//@ spec_public
	protected boolean ppbb;

	

	private boolean mb() { return true; }

	//@ spec_public
	private boolean mbb() { return true; }

	//@ spec_protected
	private boolean mbbb() { return true; }

	boolean pmb() { return true; }

	//@ spec_public
	boolean pmbb() { return true; }

	//@ spec_protected
	boolean pmbbb() { return true; }

	boolean ppmb() { return true; }

	//@ spec_public
	boolean ppmbb() { return true; }

	//@ private ghost boolean g;
	//@         ghost boolean pg;
	//@ protected ghost boolean ppg;
	//@ public ghost boolean pppg;

	//@ private model boolean mg;
	//@         model boolean pmg;
	//@ protected model boolean ppmg;
	//@ public model boolean pppmg;

// FIXME - check for model methods, model constructors, model and non-model types

	//@ ensures g && pg && ppg && pppg;
	//@ ensures mg && pmg && ppmg && pppmg;
	//@ ensures mbbb() && mbb() && mb();
	//@ ensures pmbbb() && pmbb() && pmb();
	//@ ensures ppmb() && ppmbb();
	//@ ensures bbb && bb && b;
	//@ ensures pbbb && pbb && pb;
	//@ ensures ppb && ppbb;
	public void m() {}

	//@ ensures g && pg && ppg && pppg;
	//@ ensures mg && pmg && ppmg && pppmg;
	//@ ensures mbbb() && mbb() && mb();
	//@ ensures pmbbb() && pmbb() && pmb();
	//@ ensures ppmb() && ppmbb();
	//@ ensures bbb && bb && b;
	//@ ensures pbbb && pbb && pb;
	//@ ensures ppb && ppbb;
	protected void mm() {}

}
