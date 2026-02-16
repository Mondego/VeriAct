// Test referencing superclasses via duplicated single-class import statements

// Note that this is in the unnamed package so the import is needed here.

import p.Target;
import p.Target;	// Duplication here is *not* an error
// Javac makes a mistake here and generates an error anyways...

class Single extends Target {}
