package q;

import p.Test;			// Overrides q.Test

class SingleBeatsPKG extends Test {
   // Make sure we reference p.Test, not q.Test
   int y = Test.x;	// no error
}
