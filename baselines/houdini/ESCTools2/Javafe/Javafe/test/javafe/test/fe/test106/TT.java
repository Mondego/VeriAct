/**
 ** Make sure check that a local type isn't a duplicate does not cause
 ** ambiguous typename errors.
 **
 ** See also Dual3.java, which checks the other ambiguous typename case.
 **
 ** This file must be checked with TP.java and TQ.java on the command
 ** line at the same time.
 **/


import P.*;
import Q.*;

class TT {
    void m() {
	class T {}
    }
}
