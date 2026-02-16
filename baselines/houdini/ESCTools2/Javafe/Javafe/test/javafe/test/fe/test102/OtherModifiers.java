/**
 ** Make sure no modifiers other than final are allowed on local
 ** variables or formal parameters.
 **/

class OtherModifiers {

    void params(public int p1,              // error
		private int p2,             // error
		protected int p3,           // error
		static int s,               // error
		synchronized int sy,        // error
		volatile int v,             // error
		transient int t,            // error
		native int n,               // error
		abstract int a) {           // error

	    try {}
	    catch (public Exception E) {           // error
	    } catch (private Exception E) {        // error
	    } catch (protected Exception E) {      // error
	    } catch (static Exception E) {         // error
	    } catch (synchronized Exception E) {   // error
	    } catch (volatile Exception E) {       // error
	    } catch (transient Exception E) {      // error
	    } catch (native Exception E) {         // error
	    } catch (abstract Exception E) {       // error
	    }
    }

    void locals() {
	public int v1;                  // error
	private int v2;                 // error
	protected int v2a;              // error
	static int v3;                  // error

	// continued in OtherModifiers2.java and OtherModifiers3.java ...

	transient int v6;               // error
	native int v7;                  // error
	abstract int v8;                // error
    }
}
