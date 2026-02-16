// Testing various orders of import statements, model, pragma and regular

import java.util.ArrayList;
//@ model import java.math.*;
import java.util.List;
//@ import java.util.Set;
//@ import java.util.Collection;
//@ import java.util.Vector;
import java.util.Iterator;

public class ParseImports2 {
	//@ model Collection c;
	//@ ghost Vector v;
	//@ ghost BigInteger b;
	//@ model Set s;
}
