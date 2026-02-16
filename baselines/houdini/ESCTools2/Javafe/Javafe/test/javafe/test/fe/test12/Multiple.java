// This tests the use of imports from different packages:

import p.*;
import q.*;

abstract class Multiple extends Target {}	// ref p.Target

class Multiple2 extends Q {}			// ref q.Q
