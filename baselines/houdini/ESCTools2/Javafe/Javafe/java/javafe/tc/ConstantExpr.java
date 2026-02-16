/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;


import javafe.ast.*;
import javafe.util.*;


public class ConstantExpr {

  // ----------------------------------------------------------------------

  //@ requires t != null;
  public static boolean constantValueFitsIn( Object val, PrimitiveType t ) {
    if( val instanceof Integer || val instanceof Long ) {
      long l = getLongConstant( val );
      switch( t.getTag() ) {
      case TagConstants.INTTYPE:
	return Integer.MIN_VALUE <= l && l <= Integer.MAX_VALUE;
      case TagConstants.SHORTTYPE:
	return -0x8000 <= l && l <= 0x7fff;
      case TagConstants.BYTETYPE:
	return -0x80 <= l && l <= 0x7f;
      case TagConstants.CHARTYPE:
	return 0x0 <= l && l <= 0xffff;
      default:
	return false;
      }
    }
    else
      return false;
  }
  

  /** Evaluates a compile-time constant expression.  Returns Integer,
   * Long, Float, Double, Boolean, String or null (if the expression
   * is not a constant.) 

   * The relation between the FlowInsensitiveChecks.getType(e),
   * and the type of eval(e) is as follows:

   * <PRE>
   * getType(e)          eval(e)
   *
   * boolean             Boolean or null
   * byte                Integer or null    (*)
   * short               Integer or null    (*)
   * char                Integer or null    (*)
   * int                 Integer or null
   * long                Long or null
   * float               Float or null
   * double              Double or null
   * String              String or null
   * </PRE>
   *
   * These will have been widened to int appropriately...
   */

  //@ requires e != null;
  public static Object eval(Expr e) {

    Type t = FlowInsensitiveChecks.getType( e );

    //System.out.println("eval at "+Location.toString( e.getStartLoc() )+" type "+Types.printName(t) );

    try {

      switch( e.getTag() ) {

      case TagConstants.PARENEXPR:
	return eval( ((ParenExpr)e).expr );

      case TagConstants.CONDEXPR:
	{
	  CondExpr ce = (CondExpr)e;
	  Object val = eval(ce.test);
	  if( val!= null && val instanceof Boolean ) {
	    if( ((Boolean)val).booleanValue() ) 
	      return eval(ce.thn);
	    else
	      return eval(ce.els);
	  } else
	    return null;
	}

      case TagConstants.CASTEXPR:
	{
	  CastExpr ce = (CastExpr)e;
	  Object val = eval(ce.expr);
	  if( val == null ) return null;

	  Type tsub = FlowInsensitiveChecks.getType(ce.expr);
	  if( Types.isSameType( t, tsub ) ) {
	    // Identity conversion
	    return val;
	  } 
	  else if( Types.isSameType( t, Types.javaLangString() ) ) {
	    // convert to a string
	    return val.toString();
	  } 
	  else if( Types.isIntegralType( t ) ) {
	    if( Types.isSameType( t, Types.longType ) ) {
	      if( Types.isFloatingPointType( tsub ) ) 
		return new Long( (long)getDoubleConstant( val ) );
	      else if( Types.isIntegralType( tsub ) ) 
		return new Long( getLongConstant( val ) );
	      else return null;
	    } else {
		// Type uses Integer representation:
	      
		// First, get ce.expr's val as a widened int:
		int ival;
		if (Types.isFloatingPointType(tsub)) 
		    ival = (int)getDoubleConstant(val);
		else if (Types.isIntegralType(tsub)) 
		    ival = (int)getLongConstant(val);
		else return null;

		// Do narrowing conversion if needed:
		if (Types.isByteType(t))
		    ival = (byte)ival;
		else if (Types.isShortType(t))
		    ival = (short)ival;
		else if (Types.isCharType(t))
		    ival = (char)ival;

		return new Integer(ival);
	    }
	  } else if( Types.isFloatingPointType( t ) ) {
	    if( Types.isSameType( t, Types.doubleType ) ) {
	      if( Types.isFloatingPointType( tsub ) ) 
		return new Double( getDoubleConstant( val ) );
	      else if( Types.isIntegralType( tsub ) ) 
		return new Double( getLongConstant( val ) );
	      else return null;
	    } else {
	      // Type is float
	      
	      if( Types.isFloatingPointType( tsub ) ) 
		  return new Float( (float)getDoubleConstant( val ) );
		else if( Types.isIntegralType( tsub ) ) 
		  return new Float( getLongConstant( val ) );
		else return null;
	      }
	    } else {
	      Assert.fail("Bad cast");
	      return null; // dummy return
	    }
	  }
	
      case TagConstants.FIELDACCESS: 
	{
	  FieldAccess fa = (FieldAccess)e;
	  // Assume fa.od is prepped, and decl is defined

	  VarInit init = fa.decl.init;	        //@ nowarn Null;

	  if(Modifiers.isFinal( fa.decl.modifiers )
	     && init != null 
	     && init instanceof Expr ) {

	    TypeCheck.inst.makeFlowInsensitiveChecks()
	      .checkFieldDecl( fa.decl );
	    return eval( (Expr)fa.decl.init );
	  }
	  else
	    return null;
	}

      case TagConstants.VARIABLEACCESS:
	{
	  VariableAccess lva = (VariableAccess)e;
	  if( lva.decl instanceof LocalVarDecl ) {
	    LocalVarDecl d = (LocalVarDecl)lva.decl;

	    if(Modifiers.isFinal( d.modifiers ) 
	       && d.init != null
	       && d.init instanceof Expr )
	    return eval( (Expr)d.init );
	    else
	      return null;
	  }
	  else 
	    // refers to a formal parameter
	    return null;
	}
	
      default:
	if( e instanceof LiteralExpr ) {
	  // System.out.println("eval: LiteralExpr: value="+((LiteralExpr)e).value);
	  return ((LiteralExpr)e).value;
	}
	else if( e instanceof UnaryExpr ) {

	  UnaryExpr ue = (UnaryExpr)e;
	  Object val = eval( ue.expr );
	  // System.out.println("eval: unary: op="+e.getTag()+" sub value="+val);
	  if( val==null ) return null;

	  if( Types.isSameType( t, Types.intType ) ) {
	    int i = getIntConstant( val );
	    switch( e.getTag() ) {
	    case TagConstants.UNARYADD: 
	      return val;
	    case TagConstants.UNARYSUB: 
	      return new Integer(-i); 
	    case TagConstants.BITNOT: 
	      return new Integer(~i); 
	    default:
	      return null;
	    }
	  } else if( Types.isSameType( t, Types.longType ) ) {
	    long l = getLongConstant( val );
	    switch( e.getTag() ) {
	    case TagConstants.UNARYADD: 
	      return val;
	    case TagConstants.UNARYSUB: 
	      return new Long(-l); 
	    case TagConstants.BITNOT: 
	      return new Long(~l); 
	    default:
	      return null;
	    }
	  } else if( Types.isSameType( t, Types.floatType ) ) {
	    float f = getFloatConstant( val );
	    switch( e.getTag() ) {
	    case TagConstants.UNARYADD: 
	      return val;
	    case TagConstants.UNARYSUB: 
	      return new Float(-f); 
	    default:
	      return null;
	    }
	  } else if( Types.isSameType( t, Types.doubleType ) ) {
	    double d = getDoubleConstant( val );
	    switch( e.getTag() ) {
	    case TagConstants.UNARYADD: 
	      return val;
	    case TagConstants.UNARYSUB: 
	      return new Double(-d); 
	    default:
	      return null;
	    }
 	  } else if( Types.isBooleanType( t ) ) {
	    switch( e.getTag() ) {
	    case TagConstants.NOT: 
	      return Boolean.valueOf( ! getBooleanConstant(val) );
	    default:
	      return null;
	    }
	  } else
	    return null;
	}
	else if( e instanceof BinaryExpr ) {

	  BinaryExpr be = (BinaryExpr) e;
	  Object lval = eval( be.left );
	  Object rval = eval( be.right );

	  if( lval==null || rval==null ) return null;

	  if( Types.isSameType( t, Types.intType ) )
	    return evalIntBinaryOp( e.getTag(), lval, rval );
	  else if( Types.isSameType( t, Types.longType ) )
	    return evalLongBinaryOp( e.getTag(), lval, rval );
	  else if( Types.isSameType( t, Types.floatType ) )
	    return evalFloatBinaryOp( e.getTag(), lval, rval );
	  else if( Types.isSameType( t, Types.doubleType ) )
	    return evalDoubleBinaryOp( e.getTag(), lval, rval );
	  else if( Types.isBooleanType( t ) )
	    return evalBooleanBinaryOp( e.getTag(), lval, rval );
	  else
	    return null;
	}

	return null;
      }
    } catch( ArithmeticException ex ) {
      ErrorSet.error("Arithmetic exception ("+ex.getMessage()
		     +") in evaluating constant expression");
      return null;
    } catch( AssertionFailureException ex ) {
      System.out.println("At "+Location.toString( e.getStartLoc() ));
      throw ex;
    }
  }

  // ----------------------------------------------------------------------

  private static Object evalIntBinaryOp(int op, 
					Object leftVal, 
					Object rightVal ) 
	throws ArithmeticException {

    int x = getIntConstant(leftVal);
    int y = getIntConstant(rightVal);

    switch( op ) {
    default: 
      return null;
    case TagConstants.ADD:     return new Integer(x+y);    
    case TagConstants.SUB:     return new Integer(x-y);    
    case TagConstants.STAR:    return new Integer(x*y);    
    case TagConstants.DIV:   return new Integer(x/y); //@nowarn ZeroDiv;//caught
    case TagConstants.MOD:   return new Integer(x%y); //@nowarn ZeroDiv;//caught
    case TagConstants.LSHIFT:  return new Integer(x<<y);   
    case TagConstants.RSHIFT:  return new Integer(x>>y);   
    case TagConstants.URSHIFT: return new Integer(x>>>y);  
    case TagConstants.LT:      return Boolean.valueOf(x<y);  
    case TagConstants.LE:      return Boolean.valueOf(x<=y); 
    case TagConstants.GT:      return Boolean.valueOf(x>y);  
    case TagConstants.GE:      return Boolean.valueOf(x>=y); 
    case TagConstants.EQ:      return Boolean.valueOf(x==y); 
    case TagConstants.NE:      return Boolean.valueOf(x != y); 
    case TagConstants.BITAND:  return new Integer(x&y); 
    case TagConstants.BITOR:   return new Integer(x|y); 
    case TagConstants.BITXOR:  return new Integer(x^y); 
    }
  }

  private static Object evalLongBinaryOp(int op, 
					 Object leftVal, 
					 Object rightVal )
	throws ArithmeticException {

    long x = getLongConstant(leftVal);
    long y = getLongConstant(rightVal);

    switch( op ) {
    default:
      return null;
    case TagConstants.ADD:     return new Long(x+y);    
    case TagConstants.SUB:     return new Long(x-y);    
    case TagConstants.STAR:    return new Long(x*y);    
    case TagConstants.DIV:     return new Long(x/y); //@ nowarn ZeroDiv;//caught
    case TagConstants.MOD:     return new Long(x%y); //@ nowarn ZeroDiv;//caught
    case TagConstants.LSHIFT:  return new Long(x<<y);   
    case TagConstants.RSHIFT:  return new Long(x>>y);   
    case TagConstants.URSHIFT: return new Long(x>>>y);  
    case TagConstants.LT:      return Boolean.valueOf(x<y);  
    case TagConstants.LE:      return Boolean.valueOf(x<=y); 
    case TagConstants.GT:      return Boolean.valueOf(x>y);  
    case TagConstants.GE:      return Boolean.valueOf(x>=y); 
    case TagConstants.EQ:      return Boolean.valueOf(x==y); 
    case TagConstants.NE:      return Boolean.valueOf(x != y); 
    case TagConstants.BITAND:  return new Long(x&y); 
    case TagConstants.BITOR:   return new Long(x|y); 
    case TagConstants.BITXOR:  return new Long(x^y); 
    }
  }

  private static Object evalBooleanBinaryOp(int op, 
					     Object leftVal, 
					     Object rightVal )
	throws ArithmeticException {

    if (leftVal instanceof Float || leftVal instanceof Double ||
	rightVal instanceof Float || rightVal instanceof Double) {
      return evalDoubleBinaryOp(op, leftVal, rightVal);
    } else if (! (leftVal instanceof Boolean)) {
      return evalLongBinaryOp(op, leftVal, rightVal);
    }

    boolean x = getBooleanConstant(leftVal);
    boolean y = getBooleanConstant(rightVal);

    switch( op ) {
    default:
      return null;
    case TagConstants.EQ:      return Boolean.valueOf(x==y); 
    case TagConstants.NE:      return Boolean.valueOf(x != y); 
    case TagConstants.BITAND:  return Boolean.valueOf(x&y); 
    case TagConstants.BITOR:   return Boolean.valueOf(x|y); 
    case TagConstants.BITXOR:  return Boolean.valueOf(x^y); 
    case TagConstants.AND:     return Boolean.valueOf(x&&y); 
    case TagConstants.OR:      return Boolean.valueOf(x||y); 
    }
  }
	
  private static  Object evalFloatBinaryOp(int op, 
					   Object leftVal, 
					   Object rightVal ) {
    float x = getFloatConstant(leftVal);
    float y = getFloatConstant(rightVal);

    switch( op ) {
    default:
      return null;
    case TagConstants.ADD: return new Float(x+y); 
    case TagConstants.SUB: return new Float(x-y); 
    case TagConstants.STAR:return new Float(x*y); 
    case TagConstants.DIV: return new Float(x/y); 
    case TagConstants.MOD: return new Float(x%y); 
    case TagConstants.EQ:  return Boolean.valueOf(x==y); 
    case TagConstants.NE:  return Boolean.valueOf(x != y); 
    case TagConstants.LT:  return Boolean.valueOf(x<y);  
    case TagConstants.LE:  return Boolean.valueOf(x<=y); 
    case TagConstants.GT:  return Boolean.valueOf(x>y);  
    case TagConstants.GE:  return Boolean.valueOf(x>=y); 
    }
  }

  private static  Object evalDoubleBinaryOp(int op, 
					    Object leftVal, 
					    Object rightVal ) {
    double x = getDoubleConstant(leftVal);
    double y = getDoubleConstant(rightVal);

    switch( op ) {
    default:
      return null;
    case TagConstants.ADD: return new Double(x+y); 
    case TagConstants.SUB: return new Double(x-y); 
    case TagConstants.STAR:return new Double(x*y); 
    case TagConstants.DIV: return new Double(x/y); 
    case TagConstants.MOD: return new Double(x%y); 
    case TagConstants.EQ:  return Boolean.valueOf(x==y); 
    case TagConstants.NE:  return Boolean.valueOf(x != y); 
    case TagConstants.LT:  return Boolean.valueOf(x<y);  
    case TagConstants.LE:  return Boolean.valueOf(x<=y); 
    case TagConstants.GT:  return Boolean.valueOf(x>y);  
    case TagConstants.GE:  return Boolean.valueOf(x>=y); 
    }
  }

  // ----------------------------------------------------------------------

  public static int getIntConstant(Object c) {
    if( c instanceof Integer )
      return ((Integer)c).intValue();
    else {
      Assert.fail("Bad getIntConstant");
      return 0; // dummy return
    }
  }

  public static long getLongConstant(Object c) {
    if( c instanceof Long )
      return ((Long)c).longValue();
    else if( c instanceof Integer )
      return ((Integer)c).intValue();
    else {
      Assert.fail("Bad getLongConstant: "+c);
      return 0; // dummy return
    }
  }

  private static boolean getBooleanConstant(Object c) {
    if( c instanceof Boolean )
      return ((Boolean)c).booleanValue();
    else {
      Assert.fail("Bad getBooleanConstant");
      return false; // dummy return
    }
  }

  private static float getFloatConstant(Object c) {
    if( c instanceof Integer )
      return ((Integer)c).intValue();
    else if( c instanceof Long )
      return ((Long)c).longValue();
    else if( c instanceof Float )
      return ((Float)c).floatValue();
    else {
      Assert.fail("Bad getFloatConstant");
      return 0; // dummy return
    }
  }

  private static double getDoubleConstant(Object c) {
    if( c instanceof Integer )
      return ((Integer)c).intValue();
    else if( c instanceof Long )
      return ((Long)c).longValue();
    else if( c instanceof Float )
      return ((Float)c).floatValue();
    else if( c instanceof Double )
      return ((Double)c).doubleValue();
    else {
      Assert.fail("Bad getDoubleConstant");
      return 0; // dummy return
    }
  }
}
