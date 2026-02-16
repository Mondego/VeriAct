/**
 ** Make sure no modifiers other than final are allowed on local
 ** variables or formal parameters.
 **/

class OtherModifiers {

    void locals() {
	// continued from OtherModifiers.java:

	synchronized int v4;                // parse error
	//	volative int v5;
    }
}
