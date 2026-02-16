// Testing various orders of import statements, model, pragma and regular

//@ import java.util.Vector;
import java.util.ArrayList;
//@ model import java.math.*;
import java.util.List;
//@ import java.util.Set;
//@ import java.util.Collection;

public class ParseImports {
	//@ ghost Collection c;
	//@ model Set s;
	//@ ghost Vector v;
	//@ ghost BigInteger b;
}
