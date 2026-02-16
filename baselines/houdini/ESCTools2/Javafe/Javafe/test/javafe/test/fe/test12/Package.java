// Test referencing superclasses via package import statements

// Note that this is in the unnamed package so the import is needed here.

import p.*;

// Use Package2 below to avoid clashing with imported Package class name.

class Package2 extends Target {}
