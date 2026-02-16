package s;

// This tests the use of duplicate imports:

import p.*;
import q.*;
import p.*;			// Duplication here is no error

import s.*;			// import this package without error


abstract class Multiple extends Target {}	// ref p.Target (no error)

class Multiple2 extends Q {}			// ref q.Q
