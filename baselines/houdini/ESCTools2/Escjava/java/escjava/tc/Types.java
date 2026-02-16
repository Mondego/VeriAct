/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.tc;

import javafe.ast.*;
import escjava.ast.TagConstants;
import escjava.ast.EscPrimitiveType;

import javafe.tc.*;

import javafe.util.*;

public class Types extends javafe.tc.Types
{
    static {
	inst = new Types();
    }

    public static void init() {
    }

    protected javafe.tc.TypeSig makeTypeSigInstance(String simpleName,
					/*@ non_null */ Env enclosingEnv,
					/*@ non_null */ TypeDecl decl) {
		return new escjava.tc.TypeSig(simpleName,
						enclosingEnv,
						decl);
    }

    protected javafe.tc.TypeSig makeTypeSigInstance(String[] packageName,
				/*@ non_null */ String simpleName,
				javafe.tc.TypeSig enclosingType,
				TypeDecl decl,
				CompilationUnit cu) {
	return new escjava.tc.TypeSig(packageName, simpleName,
				enclosingType, decl, cu);
    }


    public static final PrimitiveType
	anyType = EscPrimitiveType.make(TagConstants.ANY);

    public static final PrimitiveType
	typecodeType = EscPrimitiveType.make(TagConstants.TYPECODE);
    //public static Type typecodeType = javaLangClass();

    public static final PrimitiveType
	locksetType = EscPrimitiveType.make(TagConstants.LOCKSET);

    public static final PrimitiveType
	objectsetType = EscPrimitiveType.make(TagConstants.OBJECTSET);

    public static final PrimitiveType
	rangeType = EscPrimitiveType.make(TagConstants.DOTDOT);

    public static final PrimitiveType
	bigintType = EscPrimitiveType.make( TagConstants.BIGINTTYPE);

    public static final PrimitiveType
	realType = EscPrimitiveType.make( TagConstants.REALTYPE);


    public static boolean isTypeType(Type t) {
	//return t.getTag() == TagConstants.TYPECODE;
	return t.getTag() == TagConstants.TYPECODE || t.equals(javaLangClass());
    }


    public boolean isSameTypeInstance(Type t, Type tt) {
	if (isTypeType(t) && isTypeType(tt)) return true;
	return super.isSameTypeInstance(t,tt);
    }

    public boolean isCastableInstance(Type t, Type tt) {
	boolean b = super.isCastableInstance(t,tt);
	if (b) return b;
	if (t.getTag() == TagConstants.BYTETYPE &&
	        tt.getTag() == TagConstants.BIGINTTYPE) return true;
	if (t.getTag() == TagConstants.SHORTTYPE &&
	        tt.getTag() == TagConstants.BIGINTTYPE) return true;
	if (t.getTag() == TagConstants.INTTYPE &&
	        tt.getTag() == TagConstants.BIGINTTYPE) return true;
	if (t.getTag() == TagConstants.LONGTYPE &&
	        tt.getTag() == TagConstants.BIGINTTYPE) return true;
	if (t.getTag() == TagConstants.TYPECODE)
		return super.isCastableInstance(javaLangClass(),tt);
	if (tt.getTag() == TagConstants.TYPECODE)
		return super.isCastableInstance(t,javaLangClass());
	return b;
    }

    public boolean isIntegralTypeInstance(Type t){
        if ((t instanceof PrimitiveType) &&
        	((PrimitiveType)t).getTag() == TagConstants.BIGINTTYPE) return true;
        return super.isIntegralTypeInstance(t);
    }

    public boolean isNumericTypeInstance(Type t){
        if( t instanceof PrimitiveType ) {
            PrimitiveType p = (PrimitiveType)t;
            if (p.getTag() == TagConstants.BIGINTTYPE) return true;
            if (p.getTag() == TagConstants.REALTYPE) return true;
        }
        return super.isNumericTypeInstance(t);
    }
    
    public boolean isFloatingPointTypeInstance(Type t){
        if( t instanceof PrimitiveType ) {
            if (((PrimitiveType)t).tag == TagConstants.REALTYPE) return true;
        }
        return super.isFloatingPointTypeInstance(t);
    }
    
    protected boolean isWideningPrimitiveConvertableInstance( Type x, Type y ) {
        
        switch( x.getTag() ) {
        case TagConstants.BYTETYPE:
        case TagConstants.SHORTTYPE:
        case TagConstants.INTTYPE:
        case TagConstants.LONGTYPE:
            switch( y.getTag() ) {
            case TagConstants.BIGINTTYPE: 
            case TagConstants.REALTYPE:
                return true;
            default:
                break;
            }
        
        case TagConstants.BIGINTTYPE:
            switch( y.getTag() ) {
            case TagConstants.REALTYPE:
                return true;
            default:
                break;
            }  
        
        case TagConstants.FLOATTYPE:
        case TagConstants.DOUBLETYPE:
            switch( y.getTag() ) {
            case TagConstants.REALTYPE:
                return true;
            default:
                break;
            }
        default:
            break;
        }
    
    return super.isWideningPrimitiveConvertableInstance(x,y);
}



    /**
     * This routine overrides {@link javafe.tc.Types#lookupField}.  Unlike that
     * routine, it knows about ghost fields and spec_public.
     *
     * <p> This routine assumes we are in an annotation so ghost fields are visible
     * and spec_public is equivalent to public. </a>
     */
/*
    //@ requires t != null
    public FieldDecl lookupFieldInstance(Type t, Identifier id, javafe.tc.TypeSig caller) 
	    throws LookupException {
	Assert.notNull(t);
	if (t instanceof TypeName)
	    t = TypeSig.getSig((TypeName) t);
	Assert.notNull(t);

	/ *
	 * Looking up a field in an arraytype is equivalent to looking
	 * up that field in java.lang.Object unless the field name is
	 * "length":
	 * /
	if (t instanceof ArrayType && id != javafe.tc.Types.lenId)
	    t = javaLangObject();

	// Our functionality is different only for TypeSigs:
	if (!(t instanceof TypeSig))
            return javafe.tc.Types.lookupField(t, id, caller);
	TypeSig sig = (TypeSig)t;

	//	/*
	//	 * Extend caller to handle spec_public:
	//	 * /
	//	caller = new ExtendedTypeSig(caller);

	// Search for a normal field first; propogate any errors other
	// than NOTFOUND:
	try {
	    return sig.lookupField(id, caller);
	} catch (LookupException E) {
	    if (E.reason != LookupException.NOTFOUND)
		throw E;
	}

	// If not found, search for a ghost field:
	GhostEnv G = new GhostEnv(sig.getEnclosingEnv(), sig, false);
	FieldDecl decl = G.getGhostField(id.toString(), null);
	if (decl==null)
	    throw new LookupException(LookupException.NOTFOUND);

	// Make sure the ghost field is not ambiguous:
	if (G.getGhostField(id.toString(), decl) != null)
	    throw new LookupException(LookupException.AMBIGUOUS);

	// Ghost fields are always public, so no access checking required...
	// FIXME - no longer true

	return decl;
    }
*/
} // end of class Types

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */
