/**
 ** Make sure all references to local variables and formals from inner
 ** classes must refer to final variables.
 **
 ** This also ensures that the final modifier can be put on all local
 ** variables and formal parameters.
 **/

class FinalRequirements {

    void params(final int F, /*non final*/ int N) {
	final int f = 0;
	int n=0;
	int a = n;    // ok

	for (final int f1=0;;) {
	    for (int n1=0;;) {

		class TF { int a = F; }
		class TN { int a = N; }    // error

		class Tf { int a = f; }
		class Tn { int a = n; }    // error

		class Tf1 { int a = f1; }
		class Tn1 { int a = n1; }  // error

		try {}
		catch (final java.io.IOException fE) {
		    class TfE { Exception a = fE; }
		} catch (Exception nE) {
		    class TnE { Exception a = nE; }  // error
		}

		// try 2 levels deep:
		class Inner {
		    class T {
			int a = f;
			int b = n;          // error
		    }
		}
	    }
	}
    }
}
