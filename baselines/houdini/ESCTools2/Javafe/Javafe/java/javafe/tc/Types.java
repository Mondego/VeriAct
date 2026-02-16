/* $Id$
 *
 * Copyright 2000, 2001, Compaq Computer Corporation; 
 * 2006 dsrg.org.
 */

package javafe.tc;

import java.io.ByteArrayOutputStream;
import javafe.ast.*;
import javafe.tc.TagConstants;
import javafe.util.Assert;
import javafe.util.Location;

public class Types
{
    /**
     * Types uses the inst pattern to allow subclasses to provide alternative
     * implementations of some of the static methods here.
     */
    static public /*@ non_null */ Types inst;

    static {
        inst = new Types();
    }
  
    /**
     * Factory method for TypeSig structures
     */
    //@ requires !(enclosingEnv instanceof EnvForCU);
    //@ ensures \result != null;
    public static TypeSig makeTypeSig(String simpleName,
                                      /*@ non_null */ Env enclosingEnv,
                                      /*@ non_null */ TypeDecl decl) {
        return inst.makeTypeSigInstance(simpleName,
                                        enclosingEnv,
                                        decl);
    }
  
    //@ requires !(enclosingEnv instanceof EnvForCU);
    //@ ensures \result != null;
    protected TypeSig makeTypeSigInstance(String simpleName,
                                          /*@ non_null */ Env enclosingEnv,
                                          /*@ non_null */ TypeDecl decl) {
        return new javafe.tc.TypeSig(simpleName,
                                     enclosingEnv,
                                     decl);
    }

    /**
     * Factory method for TypeSig structures
     */
    //@ requires \nonnullelements(packageName);
    //@ requires (enclosingType != null) ==> (decl != null);
    //@ requires (decl==null) == (CU==null);
    //@ ensures \result != null;
    protected static TypeSig makeTypeSig(String[] packageName,
                                         /*@ non_null */ String simpleName,
                                         TypeSig enclosingType,
                                         TypeDecl decl, 
                                         CompilationUnit CU) {
        return inst.makeTypeSigInstance(packageName,
                                        simpleName,
                                        enclosingType,
                                        decl, 
                                        CU);
    }
    
    //@ requires \nonnullelements(packageName);
    //@ requires (enclosingType != null) ==> (decl != null);
    //@ requires (decl==null) == (CU==null);
    //@ ensures \result != null;
    protected TypeSig makeTypeSigInstance(String[] packageName,
                                          /*@ non_null */ String simpleName,
                                          TypeSig enclosingType,
                                          TypeDecl decl, 
                                          CompilationUnit CU) {
        return new javafe.tc.TypeSig(packageName,
                                     simpleName,
                                     enclosingType,
                                     decl, 
                                     CU);
    }
  
    // ----------------------------------------------------------------------
    // Hidden constructor

    public Types() {}

    /** Used to indicate the type of an illegal operation, so that error messages do
     * not unnecessarily propagate; should only be used if the error has already been
     * reported.
     */
    //@ invariant errorType != null;
    public static Type errorType = ErrorType.make();

    // ----------------------------------------------------------------------
    // Fields for primitive types

    /*@ requires (tag == TagConstants.BOOLEANTYPE || tag == TagConstants.INTTYPE
     || tag == TagConstants.LONGTYPE || tag == TagConstants.CHARTYPE
     || tag == TagConstants.FLOATTYPE || tag == TagConstants.DOUBLETYPE
     || tag == TagConstants.VOIDTYPE || tag == TagConstants.NULLTYPE
     || tag == TagConstants.BYTETYPE || tag == TagConstants.SHORTTYPE); */
    //@ ensures \result != null;
    private static final /*@ non_null */ PrimitiveType makePrimitiveType(int tag) {
        return JavafePrimitiveType.makeNonSyntax(tag);
    }

    //@ invariant voidType != null;
    public static /*@ non_null */ PrimitiveType 
            voidType = makePrimitiveType( TagConstants.VOIDTYPE );

    //@ invariant booleanType != null;
    public static /*@ non_null */ PrimitiveType
            booleanType = makePrimitiveType( TagConstants.BOOLEANTYPE );

    //@ invariant intType != null;
    public static /*@ non_null */ PrimitiveType
            intType = makePrimitiveType( TagConstants.INTTYPE );

    //@ invariant doubleType != null;
    public static /*@ non_null */ PrimitiveType
            doubleType = makePrimitiveType( TagConstants.DOUBLETYPE );

    //@ invariant floatType != null;
    public static /*@ non_null */ PrimitiveType
            floatType = makePrimitiveType( TagConstants.FLOATTYPE );

    //@ invariant longType != null;
    public static /*@ non_null */ PrimitiveType
            longType = makePrimitiveType( TagConstants.LONGTYPE );

    //@ invariant charType != null;
    public static /*@ non_null */ PrimitiveType
            charType = makePrimitiveType( TagConstants.CHARTYPE );

    //@ invariant nullType != null;
    public static /*@ non_null */ PrimitiveType
            nullType = makePrimitiveType( TagConstants.NULLTYPE );

    //@ invariant byteType != null;
    public static /*@ non_null */ PrimitiveType
            byteType = makePrimitiveType( TagConstants.BYTETYPE );

    //@ invariant shortType != null;
    public static /*@ non_null */ PrimitiveType
            shortType = makePrimitiveType( TagConstants.SHORTTYPE );

    public static void remakeTypes() {
            errorType = ErrorType.make();
            voidType = makePrimitiveType( TagConstants.VOIDTYPE );
            booleanType = makePrimitiveType( TagConstants.BOOLEANTYPE );
            intType = makePrimitiveType( TagConstants.INTTYPE );
            doubleType = makePrimitiveType( TagConstants.DOUBLETYPE );
            floatType = makePrimitiveType( TagConstants.FLOATTYPE );
            longType = makePrimitiveType( TagConstants.LONGTYPE );
            charType = makePrimitiveType( TagConstants.CHARTYPE );
            nullType = makePrimitiveType( TagConstants.NULLTYPE );
            byteType = makePrimitiveType( TagConstants.BYTETYPE );
            shortType = makePrimitiveType( TagConstants.SHORTTYPE );

	s_javaLangPackage = null;
	s_javaLangObject = null;
	s_javaLangError = null;
	s_javaLangException = null;
	s_javaLangThrowable = null;
	s_javaLangString = null;
	s_javaLangCloneable = null;
	s_javaLangRuntimeException = null;
	s_javaLangClass = null;
	s_javaLangSystem = null;
    }

    /***************************************************
     *                                                 *
     * Fields for java.lang types:		       *
     *                                                 *
     **************************************************/

    /**
     * Return the package java.lang as a String[] for use in calling
     * OutsideEnv.lookup[deferred].
     */
    //@ ensures \nonnullelements(\result);
    public static String[] javaLangPackage() {
	if (s_javaLangPackage==null) {
	    s_javaLangPackage = new String[2];
	    s_javaLangPackage[0] = "java";
	    s_javaLangPackage[1] = "lang";
	}

	return s_javaLangPackage;
    }
    //@ invariant s_javaLangPackage==null || \nonnullelements(s_javaLangPackage);

    //@ spec_public
    private static String[] s_javaLangPackage = null;


    /**
     * Find the TypeSig for the required package-member type
     * java.lang.T.<p>
     *
     * If the type is not found in the classpath, an error message is
     * reported via ErrorSet and an unloaded TypeSig is returned.<p>
     *
     * Precondition: the TypeSig has been initialized.<p>
     */
    //@ requires T != null;
    //@ ensures \result != null;
    public static TypeSig getJavaLang(String T) {
	return OutsideEnv.lookupDeferred(javaLangPackage(), T);
    }
  

    /*
     * NOTE: All of the following javaLangXXX routines require that
     * TypeSig be properly initialized as a precondition.
     */

    //* Returns the TypeSig for java.lang.Object.
    //@ ensures \result != null;
    public static TypeSig javaLangObject() {
        if (s_javaLangObject == null)
            s_javaLangObject = getJavaLang("Object");
        return s_javaLangObject;
    }
    private static TypeSig s_javaLangObject;

    //* Returns the TypeSig for java.lang.System.
    //@ ensures \result != null;
    public static TypeSig javaLangSystem() {
        if (s_javaLangSystem == null)
            s_javaLangSystem = getJavaLang("System");
        return s_javaLangSystem;
    }
    private static TypeSig s_javaLangSystem;

    //* Returns the TypeSig for java.lang.Error.
    //@ ensures \result != null;
    public static TypeSig javaLangError() {
        if (s_javaLangError == null)
            s_javaLangError = getJavaLang("Error");
        return s_javaLangError;
    }
    private static TypeSig s_javaLangError;

    //* Returns the TypeSig for java.lang.Exception.
    //@ ensures \result != null;
    public static TypeSig javaLangException() {
        if (s_javaLangException == null)
            s_javaLangException = getJavaLang("Exception");
        return s_javaLangException;
    }
    private static TypeSig s_javaLangException;

    //* Returns the TypeSig for java.lang.Throwable.
    //@ ensures \result != null;
    public static TypeSig javaLangThrowable() {
        if (s_javaLangThrowable == null)
            s_javaLangThrowable = getJavaLang("Throwable");
        return s_javaLangThrowable;
    }
    private static TypeSig s_javaLangThrowable;

    //* Returns the TypeSig for java.lang.String.
    //@ ensures \result != null;
    public static TypeSig javaLangString() {
        if (s_javaLangString == null)
            s_javaLangString = getJavaLang("String");
        return s_javaLangString;
    }
    private static TypeSig s_javaLangString;

    //* Returns the TypeSig for java.lang.RuntimeException.
    //@ ensures \result != null;
    public static TypeSig javaLangRuntimeException() {
        if (s_javaLangRuntimeException == null)
            s_javaLangRuntimeException =
                getJavaLang("RuntimeException");
        return s_javaLangRuntimeException;
    }
    private static TypeSig s_javaLangRuntimeException;

    //* Returns the TypeSig for java.lang.Cloneable.
    //@ ensures \result != null;
    public static TypeSig javaLangCloneable() {
        if (s_javaLangCloneable == null)
            s_javaLangCloneable = getJavaLang("Cloneable");
        return s_javaLangCloneable;
    }
    private static TypeSig s_javaLangCloneable;

    //* Returns the TypeSig for java.lang.Class
    //@ ensures \result != null;
    public static TypeSig javaLangClass() {
        if (s_javaLangClass == null)
            s_javaLangClass = getJavaLang("Class");
        return s_javaLangClass;
    }
    private static TypeSig s_javaLangClass;


    /***************************************************
     *                                                 *
     * Predicates on types:			       *
     *                                                 *
     **************************************************/

    public static boolean isErrorType(Type t) {
        return t instanceof ErrorType;
    }

    public static boolean isReferenceType(Type t) {
        return !(t instanceof PrimitiveType)
	    && !isErrorType(t);
    }

    public static boolean isReferenceOrNullType(Type t) {
        return isReferenceType(t)
            || t.getTag() == TagConstants.NULLTYPE;
    }

    private static boolean isPrimitiveType(Type t, int tag) {
        return (t instanceof PrimitiveType) && ( ((PrimitiveType)t).tag == tag);
    }

    public static boolean isByteType(Type t) {
        return isPrimitiveType( t, TagConstants.BYTETYPE );
    }

    public static boolean isBooleanType(Type t) {
        return isPrimitiveType( t, TagConstants.BOOLEANTYPE );
    }

    public static boolean isShortType(Type t){
        return isPrimitiveType( t, TagConstants.SHORTTYPE );
    }
    public static boolean isCharType(Type t){
        return isPrimitiveType( t, TagConstants.CHARTYPE );
    }
    public static boolean isDoubleType(Type t){
        return isPrimitiveType( t, TagConstants.DOUBLETYPE );
    }
    public static boolean isFloatType(Type t){
        return isPrimitiveType( t, TagConstants.FLOATTYPE );
    }
    public static boolean isIntType(Type t){
        return isPrimitiveType( t, TagConstants.INTTYPE );
    }
    public static boolean isLongType(Type t){
        return isPrimitiveType( t, TagConstants.LONGTYPE );
    }
    public static boolean isVoidType(Type t){
        return isPrimitiveType( t, TagConstants.VOIDTYPE );
    }
  
    public static boolean isNumericType(Type t){
        return inst.isNumericTypeInstance(t);
    }
    public boolean isNumericTypeInstance(Type t){
        if( t instanceof PrimitiveType ) {
            switch( ((PrimitiveType)t).tag ) {
                case TagConstants.BYTETYPE: 
                case TagConstants.SHORTTYPE: 
                case TagConstants.INTTYPE: 
                case TagConstants.LONGTYPE: 
                case TagConstants.CHARTYPE: 
                case TagConstants.FLOATTYPE: 
                case TagConstants.DOUBLETYPE: 
                    return true;
                default:
                    return false;
            }
        } else
            return false;
    }

    public static boolean isIntegralType(Type t){
        return inst.isIntegralTypeInstance(t);
    }
    
    //@ ensures \result ==> t instanceof PrimitiveType;
    public boolean isIntegralTypeInstance(Type t){
        if( t instanceof PrimitiveType ) {
            switch( ((PrimitiveType)t).tag ) {
                case TagConstants.BYTETYPE: 
                case TagConstants.SHORTTYPE: 
                case TagConstants.INTTYPE: 
                case TagConstants.LONGTYPE: 
                case TagConstants.CHARTYPE: 
                    return true;
                default:
                    return false;
            }
        } else
            return false;
    }

    public static boolean isFloatingPointType(Type t){
        return inst.isFloatingPointTypeInstance(t);
    }
    
    public boolean isFloatingPointTypeInstance(Type t){
        if( t instanceof PrimitiveType ) {
            switch( ((PrimitiveType)t).tag ) {

                case TagConstants.FLOATTYPE: 
                case TagConstants.DOUBLETYPE: 
                    return true;
                default:
                    return false;
            }
        } else
            return false;
    }

    // ======================================================================
    // Conversions on Types

    //@ requires x != null && y != null;
    /*@ ensures \result ==>
     (x instanceof PrimitiveType) == (y instanceof PrimitiveType); */
    public static boolean isSameType( Type x, Type y ) {
        return inst.isSameTypeInstance(x, y);
    }

    //@ requires x != null && y != null;
    /*@ ensures \result ==>
     (x instanceof PrimitiveType) == (y instanceof PrimitiveType); */
    protected boolean isSameTypeInstance( Type x, Type y ) {
        if( x instanceof TypeName ) x = TypeSig.getSig( (TypeName)x);
        if( y instanceof TypeName ) y = TypeSig.getSig( (TypeName)y);

        int xTag = x.getTag();
        if( xTag != y.getTag() ) return false;

        switch( xTag ) {
            case TagConstants.ARRAYTYPE:
                return isSameType( ((ArrayType)x).elemType, ((ArrayType)y).elemType );
            case TagConstants.TYPESIG:
                return x==y;
            default:
                // x and y are the same primitive type
                return true;
        }
    }

    /** Returns true if and only if <code>x</code> is a subclass or
     * superinterface of <code>y</code>.  (The occurrence of "class"
     * in the name of the method is rather unfortunate.)
     */
    //@ requires x != null && y != null;
    //@ ensures \result ==> (x instanceof TypeSig) || (x instanceof TypeName);
    public static boolean isSubclassOf( Type x, TypeSig y ) {
    
        if (x instanceof TypeName)
            x = TypeSig.getSig( (TypeName)x);

        Assert.notNull(y);

        if( x instanceof TypeSig ) {
            return ((TypeSig)x).isSubtypeOf(y);
        } else
            return false;
    }

    /** 
     * Returns true iff <code>x</code> is a superclass or
     * superinterface of <code>y</code>, or if <code>x</code> is the
     * same type as <code>y</code>.
     *
     * <b>Warning</b>: This is *not* the same as is <code>x</code> a
     * subtype of <code>y</code>!  It does not consider short below
     * int.
     */
    //@ requires x != null && y != null;
    public static boolean isSubClassOrEq(/*non_null*/ Type x,
					 /*non_null*/ Type y) {
	if (x instanceof ArrayType && y instanceof ArrayType) {
	    return isSubClassOrEq(((ArrayType)x).elemType, ((ArrayType)y).elemType);
	}

	if (x instanceof TypeName)
	    x = TypeSig.getSig((TypeName)x);

	if (y instanceof TypeName)
	    y = TypeSig.getSig((TypeName)y);

	if (x instanceof TypeSig && y instanceof TypeSig)
	    return ((TypeSig)x).isSubtypeOf((TypeSig)y);
	else
	    return isSameType(x, y);
    }


    /** Checks if one Type is castable to another.
     See JLS, P.67.
     */
  
    //@ requires s != null && t != null;
    public static boolean isCastable( Type s, Type t ) {
        // Replace TypeNames by corresponding TypeSigs
        if( s instanceof TypeName ) s = TypeSig.getSig( (TypeName)s);
        if( t instanceof TypeName ) t = TypeSig.getSig( (TypeName)t);
        return inst.isCastableInstance(s, t);
    }
  
    //@ requires s != null && t != null;
    protected boolean isCastableInstance( Type s, Type t ) {
        Assert.notNull( s );
        Assert.notNull( t );

    
        if( s instanceof PrimitiveType ) 
        {
            if( t instanceof PrimitiveType ) {
                return isAnyPrimitiveConvertable( (PrimitiveType)s, (PrimitiveType)t );
            }
            else if( s.getTag() == TagConstants.NULLTYPE ) {
                // a cast from null to a reference type
                return true;
            }
        }
        else if( s instanceof TypeSig ) 
        {
            TypeSig sSig = (TypeSig)s;
            TypeDecl sDecl = sSig.getTypeDecl();
            if( sDecl instanceof ClassDecl ) 
            {
                // s is a class
	    
                if( t instanceof TypeSig ) 
                {
                    TypeSig tSig = (TypeSig)t;
                    TypeDecl tDecl = tSig.getTypeDecl();
                    if( tDecl instanceof ClassDecl ) 
                    {
                        // t is a class
                        // must be related classes
                        return tSig.isSubtypeOf( sSig ) 
                            || sSig.isSubtypeOf( tSig );
                    } 
                    else 
                    {
                        // t is an interface
                        // Require s is not final, or s implements t
                        return !Modifiers.isFinal( sDecl.modifiers )
                            || sSig.isSubtypeOf( tSig );
                    }
                } 
                else if( t instanceof ArrayType ) 
                {
                    // t is an array type, s must be Object
                    return isSameType( sSig, javaLangObject() );
                } 
                else
                {
                    // t is a primitive type, s is a class, so not castable
                    Assert.notFalse( t instanceof PrimitiveType ); //@nowarn Pre;
                    return false;
                }
            }
            else 
            {
                // s is an interface
                if( t instanceof TypeSig ) 
                {
                    TypeSig tSig = (TypeSig)t;
                    TypeDecl tDecl = tSig.getTypeDecl();
                    if( tDecl instanceof ClassDecl ) 
                    {
                        // t is a class
                        // require t is not final, or t implements s
                        return !Modifiers.isFinal( tDecl.modifiers ) 
                            || tSig.isSubtypeOf( sSig );
                    } 
                    else
                    {
                        // t is an interface
                        // is s and t contain methods with the same signature but
                        // different return types, then an error occurs
                        // TO BE DONE
                        return true;
                    }
                } 
                else 
                {
                    // t is a primitive or array type
                    // MAYBE SHOULD ALLOW CASTING OF CLONEABLE TO ARRAY
                    Assert.notFalse( t instanceof PrimitiveType  //@ nowarn Pre;
                                     || t instanceof ArrayType );
                    return false;
                }
            }
        } 
        else if( s instanceof ArrayType ) 
        {
            // s is an array
	
            Type sElem = ((ArrayType)s).elemType;
	
            if( t instanceof TypeSig ) 
            {
                // Must be Object or Cloneable
                Type tSig = (TypeSig)t;
                return isSameType( tSig, javaLangObject() )
                    || isSameType( tSig, javaLangCloneable() );
            }
            else if( t instanceof ArrayType )
            {
                Type tElem = ((ArrayType)t).elemType;
	    
                if( sElem instanceof PrimitiveType 
                    && tElem instanceof PrimitiveType )
                {
                    // require same element type
                    return sElem.getTag() == tElem.getTag();
                }
                else if( !(sElem instanceof PrimitiveType)
                         && !(tElem instanceof PrimitiveType) )
                {
                    // require elements to be castable
                    return isCastable( sElem, tElem );
                }
                else
                    return false;
            }
            else 
            {
                Assert.notFalse( t instanceof PrimitiveType ); //@ nowarn Pre;
                return false;
            }
        }
        // Assert.fail("Fall thru2, s="+printName(s)+" t="+t+printName(t));
        return false;
    }

    //@ requires x != null && y != null;
    public static boolean isInvocationConvertable( Type x, Type y ) {
        if( x instanceof TypeName ) x = TypeSig.getSig( (TypeName)x);
        if( y instanceof TypeName ) y = TypeSig.getSig( (TypeName)y);
        return inst.isInvocationConvertableInstance(x, y);
    }
  
    //@ requires x != null && y != null;
    protected boolean isInvocationConvertableInstance( Type x, Type y ) {
        if( isSameType(x,y) ) return true;
        if( isWideningPrimitiveConvertable(x,y) ) return true;
        if( isWideningReferenceConvertable(x,y) ) return true;
        return false;
    }

    //@ requires x != null && y != null;
    protected static boolean isWideningPrimitiveConvertable( Type x, Type y ) {
        return inst.isWideningPrimitiveConvertableInstance(x,y);
    }
        
    //@ requires x != null && y != null;
    protected boolean isWideningPrimitiveConvertableInstance( Type x, Type y ) {

        switch( x.getTag() ) {
            case TagConstants.BYTETYPE:
                switch( y.getTag() ) {
                    case TagConstants.SHORTTYPE: 
                    case TagConstants.INTTYPE: case TagConstants.LONGTYPE: 
                    case TagConstants.FLOATTYPE: case TagConstants.DOUBLETYPE:
                        return true;
                    default:
                        return false;
                }
            case TagConstants.SHORTTYPE: case TagConstants.CHARTYPE:
                switch( y.getTag() ) {
                    case TagConstants.INTTYPE: case TagConstants.LONGTYPE: 
                    case TagConstants.FLOATTYPE: case TagConstants.DOUBLETYPE:
                        return true;
                    default:
                        return false;
                }
            case TagConstants.INTTYPE:
                switch( y.getTag() ) {
                    case TagConstants.LONGTYPE: 
                    case TagConstants.FLOATTYPE: case TagConstants.DOUBLETYPE:
                        return true;
                    default:
                        return false;
                }
            case TagConstants.LONGTYPE:
                switch( y.getTag() ) {
                    case TagConstants.FLOATTYPE: case TagConstants.DOUBLETYPE:
                        return true;
                    default:
                        return false;
                }
            case TagConstants.FLOATTYPE:
                switch( y.getTag() ) {
                    case TagConstants.DOUBLETYPE:
                        return true;
                    default:
                        return false;
                }
            default:
                return false;
        }
    }

    /** Returns true iff the first argument is convertable to the second
     *  argument, either through a widening primitive conversion,
     *  a narrowing primitive conversion, or the identity conversion.
     */

    protected static boolean isAnyPrimitiveConvertable( Type x, Type y ) {
        return true;
    }

    //@ requires s != null && t != null;
    protected static boolean isWideningReferenceConvertable( Type s, Type t ) {
        return inst.isWideningReferenceConvertableInstance(s, t);
    }

    //@ requires s != null && t != null;
    protected boolean isWideningReferenceConvertableInstance( Type s, Type t ) {

        if( s instanceof TypeName ) s = TypeSig.getSig( (TypeName)s);
        if( t instanceof TypeName ) t = TypeSig.getSig( (TypeName)t);
   
        if(s instanceof TypeSig 
           && t instanceof TypeSig 
           && ((TypeSig)s).isSubtypeOf( (TypeSig)t )) 
            return true;
     
        if( s.getTag() == TagConstants.NULLTYPE &&
            ( t instanceof TypeSig || t.getTag() == TagConstants.ARRAYTYPE ) )
            return true;
     
        if( s.getTag() == TagConstants.ARRAYTYPE ) {
            if( t.getTag() == TagConstants.ARRAYTYPE ) {
                Type sElem = ((ArrayType)s).elemType;
                Type tElem = ((ArrayType)t).elemType;
	
                return isSameType( sElem, tElem )
                    || isWideningReferenceConvertable(sElem,tElem);
            } 
            else if( Types.isSameType( t, javaLangObject() ) ) {
                return true;
            } 
            else 
                return false;
        }
    
        return false;
    }

    /** Returns the TypeSig for a Type x, if x denotes a class type,
     otherwise returns null. */

    //@ requires x != null;
    public static TypeSig toClassTypeSig( Type x ) {

        switch( x.getTag() ) {

            case TagConstants.TYPENAME:
                {
                    x = TypeSig.getSig( (TypeName)x);
                    // fall thru
                }
	
            case TagConstants.TYPESIG:
                {
                    TypeSig tsig = (TypeSig)x;
                    if( tsig.getTypeDecl() instanceof ClassDecl ) {
                        return tsig;
                    } else {
                        // must be an interface type
                        return null;
                    }
                }
	
            default:
                // x is a primitive type or an array type
                return null;
        }
    }
  
    // ----------------------------------------------------------------------
    // Numeric promotions
  
    //@ requires t != null;
    //@ ensures \result != null;
    public static Type unaryPromote(Type t) {
        if( isByteType(t) || isShortType(t) || isCharType(t) )
            return intType; 
        else if( isNumericType(t) )
            return t;
        else {
            Assert.fail("Not a numeric type");
            return null;		// dummy return
        }
    }

    //@ ensures \result != null;
    public static Type binaryNumericPromotion(Type x, Type y) {
        Assert.notFalse( isNumericType(x) && isNumericType(y) );	//@ nowarn Pre;
    
        if( isDoubleType(x) || isDoubleType(y) )
            return doubleType;
        else if( isFloatType(x) || isFloatType(y) )
            return floatType;
        else if( isLongType(x) || isLongType(y) )
            return longType;
        else
            return intType;
    }

    public static Type baseType(Type t) {
	if (!(t instanceof ArrayType)) return t;
	return baseType( ((ArrayType)t).elemType );
    }

    public static LiteralExpr zeroEquivalent(Type t) {
	if (isReferenceType(t)) {
	    return LiteralExpr.make(TagConstants.NULLLIT,null,Location.NULL);
	} else if (isIntType(t)) {
	    return LiteralExpr.make(TagConstants.INTLIT, new Integer(0), Location.NULL);
	} else if (isLongType(t)) {
	    return LiteralExpr.make(TagConstants.LONGLIT, new Long(0), Location.NULL);
	} else if (isBooleanType(t)) {
	    return LiteralExpr.make(TagConstants.BOOLEANLIT, Boolean.FALSE, Location.NULL);
	} else if (isDoubleType(t)) {
	    return LiteralExpr.make(TagConstants.DOUBLELIT, new Double(0), Location.NULL);
	} else if (isFloatType(t)) {
	    return LiteralExpr.make(TagConstants.FLOATLIT, new Float(0), Location.NULL);
	} else if (isShortType(t)) {
	    return LiteralExpr.make(TagConstants.SHORTLIT, new Short((short)0), Location.NULL);
	} else if (isByteType(t)) {
	    return LiteralExpr.make(TagConstants.BYTELIT, new Byte((byte)0), Location.NULL);
	} else if (isCharType(t)) {
	    return LiteralExpr.make(TagConstants.CHARLIT, new Character((char)0), Location.NULL);
	}
	System.out.println("UNSUPPORTED TYPE - zeroEquivalent " + printName(t));
	return null;
    }

    // ----------------------------------------------------------------------
    // Miscilaneous operations
  
    //@ requires x != null && y != null;
    public static boolean isSameMethodSig(MethodDecl x, MethodDecl y) {
        if( x.id != y.id ) return false;
        return isSameFormalParaDeclVec( x.args, y.args );
    }

    //@ requires x != null && y != null;
    public static boolean 
            isSameFormalParaDeclVec(FormalParaDeclVec x, FormalParaDeclVec y) {
      
        if(x.size() != y.size() ) return false;
        for( int i=0; i<x.size(); i++ ) 
            if( !isSameType( x.elementAt(i).type, y.elementAt(i).type ) )
                return false;
        return true;
    }

    //@ requires x != null && y != null;
    //@ requires x.args.count == y.args.count;
    public static boolean routineMoreSpecific( RoutineDecl x, RoutineDecl y ) {

        // should check that type containing x is invocation convertable
        // to type containing y
    
        Assert.notFalse( x.args.size() == y.args.size() );
    
        for( int i=0; i<x.args.size(); i++ )
        {
            if( !isInvocationConvertable(x.args.elementAt(i).type,
                                         y.args.elementAt(i).type ))
                return false;
        }
        return true;
    }

    // *********************************************************************


    /**
     * Is an exception a checked one?
     */
    static boolean isCheckedException(/*@ non_null */ Type E) {
	return !Types.isSubclassOf(E, Types.javaLangRuntimeException())
	    && !Types.isSubclassOf(E, Types.javaLangError());
    }


    /**
     * Is "throws <x>" a valid overriding of "throws <y>"? <p>
     *
     * Answer: Each exception E in the list <x> must be either:
     *    (a) an unchecked exception
     *    (b) a subtype of some exception in the list <y>
     */
    //@ requires x != null && y != null;
    static boolean isCompatibleRaises( TypeNameVec x, TypeNameVec y) {
        nextx:
	for (int i=0; i<x.size(); i++) {
	    TypeSig xsig = TypeSig.getSig(x.elementAt(i));

	    // Check (a):
	    if (!isCheckedException(xsig))
		continue;

	    // Check (b):
	    for (int j=0; j<y.size(); j++) {
		if (xsig.isSubtypeOf(TypeSig.getSig(y.elementAt(j))))
		    continue nextx;
	    }
	    //CF: For Houdini uses, disable this check for now
	    return false;
	}

	return true;
    }


    static boolean isCompatibleAccess( int x, int y ) {
        if( Modifiers.isPublic(y) && !Modifiers.isPublic(x) ) 
            return false;
        if(Modifiers.isProtected(y) && !Modifiers.isPublic(x)
           && !Modifiers.isProtected(x) ) 
            return false;
        if( Modifiers.isPackage(y) && Modifiers.isPrivate(x) ) 
            return false;
        return true;
    }

    //@ requires args != null;
    //@ ensures \nonnullelements(\result);
    public static Type[] getFormalParaTypes( FormalParaDeclVec args ) {
        Type[] r = new Type[ args.size() ];
        for( int i=0; i<args.size(); i++ ) 
            r[i] = args.elementAt(i).type;
        return r;
    }


    /***************************************************
     *                                                 *
     * Generating print names for Type(s):	       *
     *                                                 *
     **************************************************/

    /**
     * Returns the name of a <code>Type</code> as a
     * <code>String</code>.  The resulting name will be fully qualified
     * if the <code>Type</code> has been name resolved. <p>
     *
     * Note: <code>t</code> may safely be null.<p>
     *
     * Precondition: <code>PrettyPrint.inst</code> != null <p>
     */
    //@ ensures \result != null;
    public static String printName(Type t) {
        return inst.printNameInstance(t);
    }

    //@ ensures \result != null;
    protected String printNameInstance(Type t) {
	if (t instanceof TypeName) {
	    TypeSig sig = TypeSig.getRawSig((TypeName)t);
	    if (sig != null)
		return sig.toString();
	} else if (t instanceof ArrayType)
	    return printName(((ArrayType)t).elemType) + "[]";

	ByteArrayOutputStream result = new ByteArrayOutputStream(20);
	javafe.ast.PrettyPrint.inst.print(result, t);
	return result.toString();
    }

    /**
     * Formats an array of <code>Type</code>s as a <code>String</code>
     * containing a parenthesized list of user-readable names.  The
     * resulting names  will be fully qualified if the
     * <code>Type</code>s have been name resolved.  <p>
     *
     * Sample output: "(int, javafe.tc.TypeSig, char[])" <p>
     *
     * Precondition: <code>PrettyPrint.inst</code> != null,
     *		      <code>ts</code> != null <p>
     */
    //@ requires ts != null;
    public static String printName(Type[] ts) {
	StringBuffer s = new StringBuffer("(");

	for (int i=0; i<ts.length; i++ ) {
	    if (i != 0)
		s.append(", ");
	    s.append(printName(ts[i]));
	}

	s.append(")");
	return s.toString();
    }
  
  
    // ======================================================================
  
    protected static /*@ non_null */ Identifier lenId = Identifier.intern("length");
  
    //@ invariant lengthFieldDecl.id == lenId;
    public static /*@ non_null */ FieldDecl lengthFieldDecl
            = FieldDecl.makeInternal(Modifiers.ACC_PUBLIC|Modifiers.ACC_FINAL,
                             null,
                             lenId,
                             Types.intType,
                             null);

    //@ requires t != null && caller != null;
    //@ ensures \result != null;
    //@ ensures \result.id == id;
    public static FieldDecl lookupField(Type t, Identifier id, TypeSig caller) 
            throws LookupException
    {
	return inst.lookupFieldInstance(t, id, caller);
    }

    //@ requires t != null && caller != null;
    //@ ensures \result != null;
    //@ ensures \result.id == id;
    protected FieldDecl lookupFieldInstance(Type t, Identifier id, TypeSig caller) 
            throws LookupException
    {
	Assert.notNull(t);
	
	if( t instanceof TypeName)
            t = TypeSig.getSig( (TypeName) t );
	
	if( t instanceof TypeSig) {
            return ((TypeSig)t).lookupField(id, caller );
	} else  if( t instanceof ArrayType ) {
            if( id == lenId ) 
                return lengthFieldDecl;
            else
                // Arrays inherit all fields from java.lang.Object:
                return javaLangObject().lookupField(id, caller);
	} else if( t instanceof PrimitiveType || isErrorType(t) ) {
            throw new LookupException( LookupException.NOTFOUND );
	} else {
            Assert.fail("Unexpected type "+ t + "(" + t.getTag() + "), " + TagConstants.toString(t.getTag()));
            return null;
	}
    }
  
    //@ requires \nonnullelements(args) && caller != null;
    //@ ensures \result != null;
    //@ ensures \result.id == id;
    public static MethodDecl lookupMethod(Type t, Identifier id, 
                                          Type[] args, TypeSig caller ) 
            throws LookupException
    {
	return inst.lookupMethodInstance(t, id, args, caller);
    }

    //@ requires \nonnullelements(args) && caller != null;
    //@ ensures \result != null;
    //@ ensures \result.id == id;
    protected MethodDecl lookupMethodInstance(Type t, Identifier id, 
                                              Type[] args, TypeSig caller ) 
            throws LookupException
    {
	// Convert TypeName's to their corresponding TypeSig:
	if (t instanceof TypeName)
            t = TypeSig.getSig( (TypeName) t );

	// All array methods are methods of java.lang.Object:
	if (t instanceof ArrayType)
            t = javaLangObject();


	// Remaining cases: TypeSig, PrimitiveType, <unknown>
	if (t instanceof TypeSig)
            return ((TypeSig)t).lookupMethod(id, args, caller );

	if (!(t instanceof PrimitiveType) && !isErrorType(t))
	    Assert.fail("Unexpected type: "+t);

	throw new LookupException( LookupException.NOTFOUND );
    }
} // end of class Types

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */
