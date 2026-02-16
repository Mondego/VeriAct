/* Copyright Hewlett-Packard, 2002 */

package escjava.tc;

import javafe.ast.*;
import escjava.ast.MapType;
import escjava.ast.TagConstants;

import javafe.tc.*;

import javafe.util.*;


public class Types extends javafe.tc.Types {

  public static PrimitiveType
    anyType = PrimitiveType.make(TagConstants.ANY, Location.NULL);

  public static PrimitiveType
    typecodeType = PrimitiveType.make(TagConstants.TYPECODE, Location.NULL);

  public static PrimitiveType
    locksetType = PrimitiveType.make(TagConstants.LOCKSET, Location.NULL);

  public Types() {
      inst = this;
  }

  protected String printNameInstance(Type t) {
      if (t instanceof MapType) {
	  MapType mt = (MapType)t;
	  return printName(mt.indexType) + "->" + printName(mt.elemType);
      } else {
	  return super.printNameInstance(t);
      }
  }

    // checking if an expression of Type x can be assigned to an expression of Type y
  protected boolean isInvocationConvertableInstance( Type x, Type y ) {
      if ((x instanceof MapType) != (y instanceof MapType)) {
	  // only one of x or y is of MapType
	  return false;
      } else if (x instanceof MapType) {
	  // both x and y are MapType
	  return isMapTypeSubtypeOf((MapType)x, (MapType)y);
      } else {
	  // both x and y are not MapType
	  return super.isInvocationConvertableInstance(x, y);
      }
  }

  private boolean isMapTypeSubtypeOf(MapType x, MapType y) {
      boolean contravariance, covariance;

      // contravariant in the indexType
      // check if y.indexType is a subtype of x.indexType
      contravariance = Types.isSubClassOrEq(y.indexType, x.indexType);
      
      // covariant in the elemType
      // check if x.elemType is a subtype of y.elemType
      if (x.elemType instanceof MapType && y.elemType instanceof MapType) {	  
	  covariance = isMapTypeSubtypeOf((MapType)x.elemType, (MapType)y.elemType);
      } else {
	  covariance = Types.isSubClassOrEq(x.elemType, y.elemType);
      }

      return contravariance && covariance;
  }

    /**
     ** This routine replaces javafe.tc.Types.lookupField.  Unlike that
     ** routine, it knows about ghost fields and spec_public.
     **
     ** This routine assumes we are in an annotation so ghost fields are
     ** visible and spec_public is equivalent to public.
     **/
    //@ requires t!=null
    public static FieldDecl lookupField(Type t, Identifier id, TypeSig caller) 
	    throws LookupException {
	Assert.notNull(t);
	if (t instanceof TypeName)
	    t = TypeSig.getSig( (TypeName) t );
	Assert.notNull(t);

	/*
	 * Looking up a field in an arraytype is equivalent to looking
	 * up that field in java.lang.Object unless the field name is
	 * "length":
	 */
	if (t instanceof ArrayType && id!=javafe.tc.Types.lenId)
	    t = javaLangObject();


	// Our functionality is different only for TypeSigs:
	if (!(t instanceof TypeSig))
	   return javafe.tc.Types.lookupField(t, id, caller);
	TypeSig sig = (TypeSig)t;


	//	/*
	//	 * Extend caller to handle spec_public:
	//	 */
	//	caller = new ExtendedTypeSig(caller);


	// Search for a normal field first; propogate any errors other
	// than NOTFOUND:
	try {
	    return sig.lookupField(id, caller);
	} catch (LookupException E) {
	    if (E.reason!=LookupException.NOTFOUND)
		throw E;
	}

	// If not found, search for a ghost field:
	GhostEnv G = new GhostEnv(sig.getEnclosingEnv(), sig, false);
	FieldDecl decl = G.getGhostField(id.toString(), null);
	if (decl==null)
	    throw new LookupException( LookupException.NOTFOUND );

	// Make sure the ghost field is not ambiguous:
	if (G.getGhostField(id.toString(), decl) != null)
	    throw new LookupException( LookupException.AMBIGUOUS );

	// Ghost fields are always public, so no access checking required...

	return decl;
    }
}
