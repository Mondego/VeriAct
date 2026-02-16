/* Copyright 2000, 2001, Compaq Computer Corporation */

package houdini;

import houdini.comsock.*;
import houdini.util.*;
import java.util.*;
import java.io.*;

import escjava.prover.*;

/**
 * Performs pickling and optimization operations on sexps.
 */
class SExpOptimizer {

    /*
     * Constant atoms
     */
    final static SExp trueSexp = Atom.fromString("TRUE");
    final static SExp falseSexp = Atom.fromString("FALSE");
    final static SExp iffSexp = Atom.fromString("IFF");
    final static SExp impliesSexp = Atom.fromString("IMPLIES");
    final static SExp expliesSexp = Atom.fromString("EXPLIES");
    final static SExp orSexp = Atom.fromString("OR");
    final static SExp andSexp = Atom.fromString("AND");
    final static SExp notSexp = Atom.fromString("NOT");
    final static SExp forAllSexp = Atom.fromString("FORALL");
    final static SExp labelPosSexp = Atom.fromString("LBLPOS");
    final static SExp labelNegSexp = Atom.fromString("LBLNEG");
    final static SExp eqSexp = Atom.fromString("EQ");
   
    /*
     * Constant booleans
     */
    final static Boolean tBool = new Boolean(true);
    final static Boolean fBool = new Boolean(false);


    /**
     * Perform simple sexp opimizations.  It doesn't flatten nested or and and yet.
     */
    static SExp optimize(SExp x) throws SExpTypeError {
	if (x.isAtom()) {
	    return x;
	}
	if (x.isInteger()) {
	    return x;
	}
	SList l = x.getList();
	if (l.isEmpty()) return x;
	int n = l.length();
	SExp head = l.at(0);
	SList result = SList.make();
	if (head == orSexp) {
	    for (int i=n-1; i>=1; i--) {
		SExp e = optimize(l.at(i));
		if (e == trueSexp) {
		    return trueSexp;
		} else if (e == falseSexp) {
		} else {
		    result = result.addFront(e);
		}
	    }
	    if (result.length() == 0) {
		return falseSexp;
	    } else if (result.length() == 1) {
		return result.at(0);
	    } else {
		result = result.addFront(orSexp);
	    } 
	} else if (head == andSexp) {
	    for (int i=n-1; i>=1; i--) {
		SExp e = optimize(l.at(i));
		if (e == falseSexp) {
		    return falseSexp;
		} else if (e == trueSexp) {
		} else {
		    result = result.addFront(e);
		}
	    }
	    if (result.length() == 0) {
		return trueSexp;
	    } else if (result.length() == 1) {
		return result.at(0);
	    } else {
		result = result.addFront(andSexp);
	    } 
	} else if (head == notSexp) {
	    SExp e = optimize(l.at(1));
	    if (e == falseSexp) {
		return trueSexp;
	    } else if (e == trueSexp) {
		return falseSexp;
	    } else if (e.isList() && e.getList().at(0) == notSexp) {
		return e.getList().at(1);
	    } else {
		result = result.addFront(e).addFront(head);
	    }
	} else if (head == SExpOptimizer.forAllSexp) {
	    SExp e = optimize(l.at(n-1));
	    if (e == falseSexp) {
		return falseSexp;
	    } else if (e == trueSexp) {
		return trueSexp;
	    } else {
		if (l.at(1).getList().isEmpty()) {
		    return e;
		}
		result = result.addFront(e);
		for (int i=n-2; i>=0; i--) {
		    e = l.at(i);
		    result = result.addFront(e);
		}
	    }
	} else if (head == SExpOptimizer.labelPosSexp) {
	    SExp e = optimize(l.at(2));
	    if (e == falseSexp) {
		return falseSexp;
	    } else {
		result = result.addFront(e).addFront(l.at(1)).addFront(l.at(0));
	    }
	} else if (head == SExpOptimizer.labelNegSexp) {
	    SExp e = optimize(l.at(2));
	    if (e == trueSexp) {
		return trueSexp;
	    } else {
		result = result.addFront(e).addFront(l.at(1)).addFront(l.at(0));
	    }
	} else if (head == SExpOptimizer.eqSexp) {
	    SExp e1 = optimize(l.at(1));
	    SExp e2 = optimize(l.at(2));
	    if (e1 == e2) {
		return trueSexp;
	    } else if (e1 == trueSexp && e2 == falseSexp ||
		e1 == falseSexp && e2 == trueSexp) {
		return falseSexp;
	    } else {
		result = result.addFront(e2).addFront(e1).addFront(l.at(0));
	    }
	} else {
	    for (int i=n-1; i>=0; i--) {
		SExp e = optimize(l.at(i));
		result = result.addFront(e);
	    }
	}
	return result;
    }

    /**
     * Puts SExp in canonical form by changing implies, explies, iff into forms
     * with only and and or.
     */
    static SExp canon(SExp s) throws SExpTypeError {
	if (!s.isList()) return s;
	SList list = s.getList();
	if (list.isEmpty()) return list;
	SExp head = list.at(0);
	if (head == iffSexp) {
	    // (IFF a b) --> (and (-> a b) (-> b a))
	    Assert.notFalse(list.length() == 3);
	    SExp a = list.at(1);

	    SExp b = list.at(2);
	    return SList.make(
			      andSexp,
			      canon(SList.make(impliesSexp, a, b)),
			      canon(SList.make(impliesSexp, b, a)));
	} else if (head == impliesSexp) {
	    Assert.notFalse(list.length() == 3);
	    return SList.make(
			      orSexp,
			      SList.make(notSexp, canon(list.at(1))),
			      canon(list.at(2)));
	} else if (head == expliesSexp) {
	    Assert.notFalse(list.length() == 3);
	    return canon(
			 SList.make(
				    impliesSexp,
				    list.at(2),
				    list.at(1)));
	}
	
	SList l = SList.make();
	int n = list.length();
	for (int i = n-1; i >= 0; i--) {
	    l = l.addFront(canon(list.at(i)));
	}
	return l;
	
    }

    static public void main(String sa[]) throws Exception {
	SExp s = SList.make(
			    Atom.fromString("IMPLIES"),
			    "hello1",
			    "cow1");
	SExp s2 = SList.make(
			    Atom.fromString("IMPLIES"),
			    "hello2",
			    "cow2");

	System.out.println(s + " ==> " + canon(s));

	s = SList.make(
		       Atom.fromString("IFF"),
		       s,
		       s2);
	System.out.println(s + " ==> " + canon(s));

	s = SList.make(
		       Atom.fromString("EXPLIES"),
		       s2,
		       "cow3");
	
	System.out.println(s + " ==> " + canon(s));

	{
	    ByteArrayOutputStream os = new ByteArrayOutputStream();
	    DataOutputStream out = new DataOutputStream(os);
	    writeToFile(s, out);
	    os.close();
	    String st = os.toString();
	    DataInputStream in = new DataInputStream(new StringBufferInputStream(st));
	    SExp t =readFromFile(in);
	    System.out.println(t + " ==> " + canon(t));
	}

	s = SList.make("OR", "1", "TRUE");
	System.out.println(s + " =opt=> " + optimize(s));

	s = SList.make("AND", "1", "TRUE");
	System.out.println(s + " =opt=> " + optimize(s));

	s = SList.make("AND", "1", "FALSE");
	System.out.println(s + " =opt=> " + optimize(s));

	s = SList.make("AND", s, s);
	System.out.println(s + " =opt=> " + optimize(s));

	s = SList.make("AND", s, "FALSE");
	System.out.println(s + " =opt=> " + optimize(s));

	s = SList.make("AND", s, SList.make("OR", "TRUE", "FALSE"));
	System.out.println(s + " =opt=> " + optimize(s));

	s = SList.make("FORALL", SList.make(), SList.make("OR", "TRUE", "FALSE"));
	System.out.println(s + " =opt=> " + optimize(s));

	s = SList.make("FORALL", SList.make("A"), SList.make("OR", "A", "FALSE"));
	System.out.println(s + " =opt=> " + optimize(s));

	s = SList.make("NOT", SList.make("OR", "TRUE", "FALSE"));
	System.out.println(s + " =opt=> " + optimize(s));

	s = SList.make("NOT", SList.make("AND", "TRUE", "FALSE"));
	System.out.println(s + " =opt=> " + optimize(s));

	s = SList.make("NOT", SList.make("NOT", "A"));
	System.out.println(s + " =opt=> " + optimize(s));


    }


    /**
     * Write an SExp to a stream in a form that does not require parsing.
     */
    public static void writeToFile(SExp x, DataOutputStream oos) throws IOException, SExpTypeError {
	if (x.isAtom()) {
	    oos.writeInt(-1);
	    String s = x.getAtom().toString();
	    oos.writeInt(s.length());
	    oos.writeBytes(s);
	}
	if (x.isInteger()) {
	    oos.writeInt(-2);
	    oos.writeInt(x.getInteger());
	}
	if (x.isList()) {
	    SList l = x.getList();
	    int n = l.length();
	    oos.writeInt(0);
	    oos.writeInt(n);
	    for (int i=n-1; i>=0; i--) {
		writeToFile(l.at(i), oos);
	    }
	}
    }

    
    private static byte id[] = new byte[1024];

    /**
     * Read in an Sexp written with the above method.
     */
    public static SExp readFromFile(DataInputStream ios) throws IOException, SExpTypeError {
	int value = ios.readInt();
	SExp result = null;
	switch (value) {
	case -1:
	    int len = ios.readInt();
	    ios.read(id, 0, len);
	    String s = new String(id, 0, len);
	    Atom a = Atom.fromString(s);
	    result = a;
	    break;
	case -2:
	    result = SInt.make(ios.readInt());
	    break;
	case 0:
	    int n = ios.readInt();
	    SList l = SList.make();
	    for (int i=0; i<n; i++) {
		l = l.addFront(readFromFile(ios));
	    }
	    result = l;
	    break;
	default:
	    System.out.println(value);
	    Assert.fail("bad id " + value);
	}
	//	System.out.println(result);
	return result;
    }

    
    /**
     * Set any occurance of a guard appearing in the table to its truth value.
     */
    static public SExp modifyGuards(SExp exp, Hashtable table) {
        SExp result = null;
        try {
            if (exp.isInteger()) {
                result =  exp;
            } else if (exp.isAtom()) {
                Boolean b = (Boolean)table.get(exp.toString());
                if (b == null) {
                    result = exp;
                } else {
                    result = b.booleanValue()?
			SExpOptimizer.trueSexp:
			SExpOptimizer.falseSexp;
                }
            } else {
                SList list = (SList)exp;
                SList resultList = SList.make();
		int n = list.length();
                for (int i = n-1; i >= 0; i--) {
                    resultList = resultList.addFront(modifyGuards(list.at(i), table));
                }
                result = resultList;
            }
        } catch (SExpTypeError e) {
            Assert.fail(e);
        }
        return result;
    }

    /**
     * Same as above, but does it in place.
     */
    static public void modifyGuardsInPlace(SExp exp, Hashtable table) {
        try {
	    if (exp.isAtom()) {
		Assert.fail("can't in place modify an atom");
	    }
            if (exp.isInteger()) {
		
            } else {
                SList list = (SList)exp;
		int n = list.length();
                for (int i = 0; i < n; i++) {
		    SExp s = list.at(i);
		    if (s.isAtom()) {
			Boolean b = (Boolean)table.get(s.toString());
			if (b != null) {
			    list.setAt(i, 
				       b.booleanValue()?
				       SExpOptimizer.trueSexp:
				       SExpOptimizer.falseSexp);
			}
		    } else {
			modifyGuardsInPlace(list.at(i), table);
		    }
		}
	    }
        } catch (SExpTypeError e) {
            Assert.fail(e);
        }
    }


}
