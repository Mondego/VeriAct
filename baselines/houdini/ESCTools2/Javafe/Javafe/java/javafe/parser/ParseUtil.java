/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.parser;

//alx: 
import javafe.Tool;
//alx-end
import javafe.ast.*;
import javafe.util.StackVector;
import javafe.util.ErrorSet;
import javafe.util.Location;

/**
 * Base class for Java parser; provides some basic parsing utilities.
 *
 * @see javafe.ast.ASTNode
 * @see javafe.parser.ParseType
 * @see javafe.parser.Parse
 */

public class ParseUtil
{
    //alx: dw if to use the universe type system and with which level
    //alx: TODO what specs for these fields
    public int universeLevel=23;
    public boolean useUniverses=false;
    // used by parseModifiers() to remember universe modifiers
    public int[] universeArray = {0,0};
    //alx-end

    public ParseUtil() {
	//@ set seqModifierPragma.elementType = \type(ModifierPragma);
	//@ set seqModifierPragma.owner = this;
    	//alx: dw init universe vars
    	if (Tool.options!=null) {
    		universeLevel=Tool.options.universeLevel;
    		useUniverses=Tool.options.useUniverseTypeSystem;
    	}
	//alx-end
    }


    //----------------------- Misc Helper Functions ----------------------------

    /** Raises a <TT>RuntimeException</TT> with the argument string. */

    //! ensures false;
    // UNUSED private static void fail(String m) { ErrorSet.fatal(m); }

    /** Raises a <TT>RuntimeException</TT> with the argument string
        including a textual representation of the given source location. */
    //@ requires loc != Location.NULL;
    //@ ensures false;
    public static void fail(int loc, String m) { ErrorSet.fatal(loc, m); }

    //@ requires loc != Location.NULL;
    public static void error(int loc, String m) { ErrorSet.error(loc, m); }

    /** 
     Takes an expected token from the input stream, 
     calls <TT>fail</TT> on error.
     */
    //@ requires l != null && l.m_in != null;
    //@ modifies l.ttype, l.auxVal, l.identifierVal;
    //@ ensures \old(l.ttype)==expected;
    public void expect(Lex l, int expected) {
        if( l.ttype != expected ) 
            fail(l.startingLoc,
                 "Unexpected token '" + PrettyPrint.inst.toString(l.ttype) +
                 "', expected '" + PrettyPrint.inst.toString(expected)+"'");
        l.getNextToken();
    }

    /**

     Converts operator tokens to corresponding AST tag.
     Tokens INC and DEC get mapped to PREFIXINC and PREFIXDEC

     */

    int operatorTokenToTag(int token) {
        // Right now we have the identity mapping.
        return token;
    }


    /*------------------------ Modifiers -------------------------------*/

    //* Internal working storage for parse*Modifier* functions
    //@ invariant seqModifierPragma.elementType == \type(ModifierPragma);
    //@ invariant seqModifierPragma.owner == this;
    protected final StackVector seqModifierPragma = new StackVector();

    /**
     * Keyword at index i in this array corresponds to bit i in
     * modifier bitset.  Thus PRIVATE is at index 1, ACC_PRIVATE =
     * 1<<1.
     * These are in the same bit order as the values in 
     * java.lang.reflect.Modifier, and should remain that way, though
     * I don't know if any code uses that fact (it might come in handy
     * in reading class files, for example).
     */
    public static final int modifierKeywords[] = {
	ParserTagConstants.PUBLIC, ParserTagConstants.PRIVATE, ParserTagConstants.PROTECTED, 
	ParserTagConstants.STATIC, ParserTagConstants.FINAL, 
	ParserTagConstants.SYNCHRONIZED, ParserTagConstants.VOLATILE,
	ParserTagConstants.TRANSIENT, ParserTagConstants.NATIVE, 
	-1, // Don't consider 'interface' to be a modifier
	ParserTagConstants.ABSTRACT, ParserTagConstants.STRICT,
	-1,-1,-1,-1
    };


    /**
     * Parse a list of modifier pragmas.  Returns <code>null</code> if
     * <code>l</code> does not point to a modifier pragma.  Otherwise,
     * reads <code>l</code> until there are no more modifier pragmas and
     * returns the resulting list.
     */
    //@ requires l.m_in != null;
    public ModifierPragmaVec parseModifierPragmas(/*@ non_null @*/ Lex l) {
	if (l.ttype != ParserTagConstants.MODIFIERPRAGMA)
	    return null;

	seqModifierPragma.push();
	do {
	    seqModifierPragma.addElement(l.auxVal);
	    l.getNextToken();
	} while (l.ttype == ParserTagConstants.MODIFIERPRAGMA);
	return ModifierPragmaVec.popFromStackVector(seqModifierPragma);
    }

    /**
     * Parse a list of modifier pragmas and adds them to an existing
     * <code>ModifierPragmaVec</code>.  If the existing
     * <code>ModifierPragmaVec</code> was <code>null</code>, then it
     * either returns <code>null</code> (if <code>l</code> does not point
     * to a modifier pragma), or returns a new
     * <code>ModifierPragmaVec</code>.
     */
    //@ requires l.m_in != null;
    public ModifierPragmaVec parseMoreModifierPragmas(/*@ non_null @*/ Lex l, 
						      ModifierPragmaVec orig)
    {
	ModifierPragmaVec modifierPragmas = parseModifierPragmas( l );
	if (modifierPragmas == null)
	    return orig;
	else if (orig == null)
	    return modifierPragmas;
	else {
	    orig.append( modifierPragmas );
	    return orig;
	}
    }

    /** As a side effect, <code>parseModifiers</code> mutates this
     value. */
    public ModifierPragmaVec modifierPragmas;


    /**
     * Parse a list of modifiers.  Ensures no duplicate Java modifiers
     * and only one of the access modifiers public, protected,
     * private.  Return integer encoding the Java modifiers.
     *
     * <p> In addition to parsing Java modifiers, also handles modifier
     * pragmas (anything with a ttype of TagConstants.MODIFIERPRAGMA).
     * If no modifier pragmas are seen, sets
     * <c>modifierPragmas</c> to <c>null</c>.  Otherwise, sets it to
     * be the list of modifier pragmas seen in the course of parsing any
     * Java modifiers.
     *
     * @see javafe.ast.Modifiers
     */
    //@ requires l.m_in != null;
    //@ modifies modifierPragmas;
    public int parseModifiers(/*@ non_null @*/ Lex l) {
        boolean seenPragma = false;
    	//alx: dw next 3 variables are used for the universe type system
    	int nof_universeModifiers=0;
    	universeArray[0]=0;
    	universeArray[1]=0;

        int modifiers = Modifiers.NONE;

        getModifierLoop:
        for(;;) {
            if (l.ttype == ParserTagConstants.MODIFIERPRAGMA) {
		//alx: dw
		//handle universe modifiers returned from PragmaParser
		if (universeLevel%2==0 && l.auxVal instanceof ModifierPragma) {
		    int tag = ((ModifierPragma) l.auxVal).getTag();
		    if (tag==ParserTagConstants.PEER || 
			tag==ParserTagConstants.REP || 
			tag==ParserTagConstants.READONLY) {
			if (nof_universeModifiers>1) {
			    if (universeLevel%5!=0)
				ErrorSet.error(((ModifierPragma) l.auxVal).getStartLoc(),
					       "too many Universe type modifiers");
			    l.getNextToken();
			    continue getModifierLoop;
			}
			universeArray[nof_universeModifiers++]=tag;
			l.getNextToken();
			continue getModifierLoop;
		    }    
		}
		//alx-end
                if (! seenPragma) { 
			seqModifierPragma.push(); 
			seenPragma = true; 
		}
                seqModifierPragma.addElement(l.auxVal);
                l.getNextToken();
                continue getModifierLoop;
            } else {
		//alx: dw handle universe modifiers that are used as keywords
		if (universeLevel%3==0 && (l.ttype==ParserTagConstants.PEER || 
					   l.ttype==ParserTagConstants.REP || 
					   l.ttype==ParserTagConstants.READONLY)) {
		    if (nof_universeModifiers>1) {
			if (universeLevel%5!=0)
			    ErrorSet.error(l.startingLoc,
					   "too many Universe type modifiers");
			l.getNextToken();
			continue getModifierLoop;
		    }
		    universeArray[nof_universeModifiers++]=l.ttype;
		    l.getNextToken();
		    continue getModifierLoop;
		}
		//alx-end
		int i = getJavaModifier(l,modifiers);
		if (i != 0) {
                    modifiers |= i;
                    continue getModifierLoop;
                }
            }

            // Next token is not a modifier
            if (! seenPragma)
                modifierPragmas = null;
            else
                modifierPragmas
                    = ModifierPragmaVec.popFromStackVector(seqModifierPragma);
            return modifiers;
        }
    }

    /** Checks if the next token is a Java modifier.  Returns 0
	if it is not; returns an int with the modifier bit turned on
	if it is.  Also issues an error if the modifier is already 
	present in the modifiers argument (use 0 for this argument
	to turn off these errors).
	Note that the bit that is found is not ORed into the modifiers
	int; you have to do that outside this call.
	The Lex is advanced if a modifier is found.
    */
    public int getJavaModifier(Lex l, int modifiers) {
	for( int i=0; i<modifierKeywords.length; i++ ) {
	    if( l.ttype == modifierKeywords[i] ) {
		// Token is modifier keyword 
		int modifierBit = 1<<i;
		if( (modifiers & modifierBit) != 0 ) {
		    ErrorSet.caution(l.startingLoc, 
			"Duplicate occurrence of modifier '"
			 +PrettyPrint.inst.toString(l.ttype)+"'");
		} else if( (modifiers & Modifiers.ACCESS_MODIFIERS) != 0 &&
		    (modifierBit & Modifiers.ACCESS_MODIFIERS) != 0 ) {
		    ErrorSet.error(l.startingLoc, 
			 "Cannot have more than one of the access modifiers "+
			 "public, protected, private");
		}
		l.getNextToken();
		return modifierBit;
	    }
	}
	return 0;
    }

    public boolean isJavaModifier(int ttype) {
	for( int i=0; i<modifierKeywords.length; i++ ) {
	    if( ttype == modifierKeywords[i] ) return true;
	}
	return false;
    }

    static public String arrayToString(Object[] a, String sep) {
	if (a==null || a.length == 0) return "";
	else {
		StringBuffer sb = new StringBuffer();
		sb.append(a[0].toString());
		for (int i=1; i<a.length; ++i) {
		    sb.append(sep);
		    sb.append(a[i].toString());
	        }
		return sb.toString();
	}
    }

    //alx: dw parses universes and rejects all other modifiers!!
    //TODO: what other specs?
    public void parseUniverses(/*@ non_null @*/ Lex l) {
    	int loc = l.startingLoc;
    	int jmods = parseModifiers(l);
    	if (jmods!=0)
    		ErrorSet.error(loc,"no java modifiers allowed");
    	if (modifierPragmas!=null && modifierPragmas.size()!=0)
    		ErrorSet.error(loc,"only Universe type modifiers allowed");
    }
    //alx-end
    
    //alx: dw using decorations to save the universe modifiers
    private static ASTDecoration universeDecoration =
	new ASTDecoration("universeDecoration");
    //alx-end
    
    //alx: dw sets the universeDecoration of node i
    //TODO: what other specs?
    public static /*@ non_null @*/ ASTNode setUniverse(
		   /*@ non_null @*/ ASTNode i, 
	           int u) {
        universeDecoration.set(i,new Integer(u));
    	return i;
    }
    //alx-end
    
    //alx: dw sets the universe modifier and array element modifier of i to
    //   the values in the array. with error-checking!
    //TODO: what other specs?
    public static /*@ non_null @*/ ASTNode setUniverse(
				      /*@ non_null @*/ ASTNode i, 
				      int[] a, 
				      /*@ non_null @*/ Type t, int loc) {
	int tag = t.getTag();
	boolean reportErrors=Tool.options!=null && 
	                     Tool.options.universeLevel%5!=0;
	if (a!=null && !(tag==ParserTagConstants.ARRAYTYPE)) {
	    if (t instanceof PrimitiveType) {
		if (a[0]!=0 && reportErrors)
		    ErrorSet.error(loc,
			   "primitive types must not have Universe modifiers");
		return i;
	    }
	    if (a[0]!=0)
		universeDecoration.set(i,new Integer(a[0]));
	    else
		universeDecoration.set(i,new Integer(ParserTagConstants.IMPL_PEER));
	    if (a[1]!=0 && reportErrors)
		ErrorSet.error(loc,
		     "only array types can have 2 Universe modifiers");
	} 
	else if (a!=null && tag==ParserTagConstants.ARRAYTYPE) {
	    ArrayType at = (ArrayType ) t;
	    //primitive types need only the array modifier
	    if (at.elemType instanceof PrimitiveType) {
		if (a[0]==0) {
		    universeDecoration.set(i,
					 new Integer(ParserTagConstants.IMPL_PEER));
		    return i;
		}
		else if (a[1]!=0 && reportErrors) 
		    ErrorSet.error(loc,
		         "only one Universe modifier allowed for "+
			 "arrays of primitve type");
		universeDecoration.set(i,new Integer(a[0]));
	    } 
	    else {
		if (a[0]==0) {
		    universeDecoration.set(i,
				      new Integer(ParserTagConstants.IMPL_PEER));
		    elementUniverseDecoration.set(i,
				      new Integer(ParserTagConstants.IMPL_PEER));
		    return i;
		}
		else if(a[1]==0) {
		    if (a[0]==ParserTagConstants.REP) {
			if (reportErrors)
			    ErrorSet.error(loc,
				 "rep not allowed for array elements");
			a[0]=ParserTagConstants.IMPL_PEER;
		    }
		    universeDecoration.set(i,
			    new Integer(ParserTagConstants.IMPL_PEER));
		    elementUniverseDecoration.set(i,new Integer(a[0]));
		} else {
		    if (a[1]==ParserTagConstants.REP) {
			if (reportErrors)
			    ErrorSet.error(loc,
				 "rep not allowed for array elements");
			a[1]=ParserTagConstants.IMPL_PEER;
		    }
		    universeDecoration.set(i,new Integer(a[0]));
		    elementUniverseDecoration.set(i,new Integer(a[1]));
		}
	    }
	}
	if (reportErrors && getUniverse(i)==ParserTagConstants.READONLY &&
	    (i.getTag()==ParserTagConstants.NEWINSTANCEEXPR
	     || i.getTag()==ParserTagConstants.NEWARRAYEXPR))
	    ErrorSet.error(i.getStartLoc(),
                 "readonly not allowed for new expression, except for "+
		 "array element modifier");
	return i;
    }
    //alx-end
    
    //alx: dw sets the universe modifier and array element modifier of i to
    //   the values in the array. with error-checking!
    //TODO: what other specs?
    /*@ 
      @ requires i.type !=null;
      @*/
    public static /*@ non_null @*/ ASTNode setUniverse(
				      /*@ non_null @*/ GenericVarDecl i, 
				      int[] a) {
    	return setUniverse(i,a,i.type,i.getStartLoc());
    }
    
    //alx: dw gets the universeDecoration of node i
    //TODO: what other specs?
    public static int getUniverse(/*@ non_null @*/ ASTNode i) {
    	Object o = universeDecoration.get(i);
    	if (o instanceof Integer)
    		return ((Integer) o).intValue();
    	return 0;
    }
    //alx-end
    
    //alx: dw using decorations for array element universe modifier
    //TODO: what specs?
    private static ASTDecoration elementUniverseDecoration
		= new ASTDecoration("elementUniverseDecoration");
    //alx-end
    
    //alx: dw sets the elementUniverseDecoration of node i
    //TODO: what other specs?
    public static /*@ non_null @*/ ASTNode setElementUniverse(
					     /*@ non_null @*/ ASTNode i, 
					     int u) {
    	elementUniverseDecoration.set(i,new Integer(u));
    	return i;
    }
    //alx-end
    
    //alx: dw gets the elementUniverseDecoration of node i
    //TODO: what other specs?
    public static int getElementUniverse(/*@ non_null @*/ ASTNode i) {
    	Object o = elementUniverseDecoration.get(i);
    	if (o instanceof Integer)
    		return ((Integer) o).intValue();
    	return 0;
    }
    //alx-end
}
