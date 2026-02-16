// $Id$
/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;

import java.util.HashSet;
import javafe.ast.ASTDecoration;
import javafe.ast.ASTNode;
import javafe.ast.AmbiguousMethodInvocation;
import javafe.ast.AmbiguousVariableAccess;
import javafe.ast.ArrayInit;
import javafe.ast.ArrayRefExpr;
import javafe.ast.ArrayType;
import javafe.ast.AssertStmt;
import javafe.ast.BinaryExpr;
import javafe.ast.BranchStmt;
import javafe.ast.CastExpr;
import javafe.ast.CatchClause;
import javafe.ast.ClassDecl;
import javafe.ast.ClassDeclStmt;
import javafe.ast.ClassLiteral;
import javafe.ast.CondExpr;
import javafe.ast.ConstructorDecl;
import javafe.ast.ConstructorInvocation;
import javafe.ast.DoStmt;
import javafe.ast.EvalStmt;
import javafe.ast.Expr;
import javafe.ast.ExprObjectDesignator;
import javafe.ast.ExprVec;
import javafe.ast.FieldAccess;
import javafe.ast.FieldDecl;
import javafe.ast.ForStmt;
import javafe.ast.FormalParaDecl;
import javafe.ast.GenericBlockStmt;
import javafe.ast.GenericVarDecl;
import javafe.ast.Identifier;
import javafe.ast.IfStmt;
import javafe.ast.InitBlock;
import javafe.ast.InstanceOfExpr;
import javafe.ast.InterfaceDecl;
import javafe.ast.LabelStmt;
import javafe.ast.LocalVarDecl;
import javafe.ast.MethodDecl;
import javafe.ast.MethodInvocation;
import javafe.ast.ModifierPragma;
import javafe.ast.ModifierPragmaVec;
import javafe.ast.Modifiers;
import javafe.ast.Name;
import javafe.ast.NewArrayExpr;
import javafe.ast.NewInstanceExpr;
import javafe.ast.ObjectDesignator;
import javafe.ast.ParenExpr;
import javafe.ast.PrettyPrint;
import javafe.ast.PrimitiveType;
import javafe.ast.ReturnStmt;
import javafe.ast.RoutineDecl;
import javafe.ast.SkipStmt;
import javafe.ast.Stmt;
import javafe.ast.StmtPragma;
import javafe.ast.StmtVec;
import javafe.ast.SuperObjectDesignator;
import javafe.ast.SwitchLabel;
import javafe.ast.SwitchStmt;
import javafe.ast.SynchronizeStmt;
import javafe.ast.ThisExpr;
import javafe.ast.ThrowStmt;
import javafe.ast.TryCatchStmt;
import javafe.ast.TryFinallyStmt;
import javafe.ast.Type;
import javafe.ast.TypeDecl;
import javafe.ast.TypeDeclElem;
import javafe.ast.TypeDeclElemPragma;
import javafe.ast.TypeModifierPragma;
import javafe.ast.TypeModifierPragmaVec;
import javafe.ast.TypeName;
import javafe.ast.TypeObjectDesignator;
import javafe.ast.UnaryExpr;
import javafe.ast.VarDeclStmt;
import javafe.ast.VarInit;
import javafe.ast.VariableAccess;
import javafe.ast.WhileStmt;
import javafe.parser.ParseUtil;
import javafe.util.Assert;
import javafe.util.ErrorSet;
import javafe.util.Location;

/**
 * Does disambiguation and flow insensitive checks on a type
 * declaration.
 */
// TODO: Universes should be implemented by subclassing (rgrig).
public class FlowInsensitiveChecks
{
    static public FlowInsensitiveChecks inst = new FlowInsensitiveChecks();
    static public FlowInsensitiveChecks inst() { return inst; }

    /**
     * Controls whether or not implicit super-calls in constructors
     * are made explicit.  By default they are.
     */
    public static boolean dontAddImplicitConstructorInvocations = false;

    //alx: fields introduced by dw
    protected boolean useUniverses=false;
    boolean readonlyStdForPureCtor=false;
    boolean inPure=false;
    boolean inCtor=false;
    boolean inStatic=false;
    int universeReturnType = 0;
    int universeElementReturnType = 0;
    //alx-end

    // Use this to get an instance in order to get proper derived behavior.
    protected FlowInsensitiveChecks() { 
	//alx: dw init useUniverses
    	useUniverses=javafe.Tool.options!=null && 
                     javafe.Tool.options.useUniverseTypeSystem && 
                     javafe.Tool.options.universeLevel%5!=0;
    	readonlyStdForPureCtor=javafe.Tool.options!=null && 
	             javafe.Tool.options.useUniverseTypeSystem && 
                     javafe.Tool.options.universeLevel%7==0;
    }

    /**
     * Factory method so subclasses can override.
     */
    //@ requires s != null;
    //@ ensures \result != null;
    protected EnvForTypeSig makeEnvForTypeSig(TypeSig s,
					      boolean staticContext) {
	return s.getEnv(staticContext);
    }


    ///////////////////////////////////////////////////////////////////////
    //                                                                   //
    // Information that remains the same during processing of entire     //
    // type decl.							 //
    //                                                                   //
    ///////////////////////////////////////////////////////////////////////

    //@ invariant sig != null ==> rootIEnv != null && rootSEnv != null;
    /*@ spec_public */ protected TypeSig sig;

    // Inv: rootIEnv.peer == sig; !rootIEnv.staticContext
    /*@ spec_public */ protected EnvForTypeSig rootIEnv;

    // Inv: rootIEnv.peer == sig; !rootIEnv.staticContext
    /*@ spec_public */ protected EnvForTypeSig rootSEnv;
  

    ///////////////////////////////////////////////////////////////////////
    //                                                                   //
    // Information that changes for the processing of each member of     //
    // the decl.							 //
    //                                                                   //
    ///////////////////////////////////////////////////////////////////////
  
    //* Cannot access fields defined later? 
    protected boolean leftToRight;

    //* Type for return statements. 
    protected Type returnType;
  

    ///////////////////////////////////////////////////////////////////////
    // Information that changes within a member in a stack-like          //
    // manner.                                                           //
    ///////////////////////////////////////////////////////////////////////
  
    //@ invariant allowedExceptions != null;
    protected TypeSigVec allowedExceptions = TypeSigVec.make();

    //@ invariant enclosingLabels != null;
    protected StmtVec enclosingLabels = StmtVec.make();

    // -------------------------------------------------------------
    /**
     * Moves <code>s</code> into implementation checked state.
     *
     */
    //@ requires (* <code>s</code> is in prepped state.*);
    //@ requires s.state >= TypeSig.PREPPED;
    public void checkTypeDeclaration(/*@ non_null @*/ TypeSig s) {
        Assert.precondition(s.state >= TypeSig.PREPPED);

        // Set up variables for entire traversal:
        rootSEnv = makeEnvForTypeSig(s, true);
        rootIEnv = makeEnvForTypeSig(s, false);
        sig = s;

        TypeDecl d = s.getTypeDecl();

        // Process each member declaration
        for(int i = 0, sz = d.elems.size(); i < sz; i++) {
            TypeDeclElem e = d.elems.elementAt(i);
            checkTypeDeclElem(e);
        }

        // Process ModifierPragmas
        checkModifierPragmaVec(d.pmodifiers, d, rootSEnv);

	sig = null;
    }

    // -------------------------------------------------------------

    /**
     * Moves <code>fd</code> into implementation checked state.
     *
     */
    //@ requires (* <code>fd</code> is in prepped state. *);
    //@ modifies sig, rootSEnv, rootIEnv;
    public void checkFieldDecl(/*@ non_null @*/ FieldDecl fd) {
	/*
	 * Special case for free-floating fields like length.
	 *
	 * Such fields must not have any modifier pragmas or an initializer.
	 */
	if (fd.parent==null) {
	    Assert.notFalse(fd.pmodifiers == null,   //@ nowarn Pre;
                            "Free-floating FieldDecls may not have any ModifierPragmas");
	    Assert.notFalse(fd.init == null,         //@ nowarn Pre;
                            "Free-floating FieldDecls may not have initializers");

	    return;         // No other works needs to be done
	}


	TypeSig sig = TypeSig.getSig(fd.parent);
	if (sig.state < TypeSig.CHECKED) {
	    // Type check this decl
	
	    // Info.out("[Pre-checking "+Types.printName(sig)+"."+fd.id+"]");
	    sig.prep();

	    // Set up variables for entire traversal
	    rootSEnv = makeEnvForTypeSig(sig, true);
	    rootIEnv = makeEnvForTypeSig(sig, false);
	    this.sig = sig;

	    checkTypeDeclElem(fd);
	}
	this.sig = null;
    }

    /**
     * Tests if universe modifiers of <code>r</code> can be 
     * assigned to <code>l</code>.
     *
     */
    //alx: dw test if universe modifiers of r can be assigned to l
    //TODO: check if the specs are enough
    public static void checkUniverseAssignability(/*@ non_null @*/ ASTNode l, 
						  /*@ non_null @*/ ASTNode r) {
	if (!universeIsSubtypeOf(ParseUtil.getUniverse(r),
                                 ParseUtil.getUniverse(l)))
	    ErrorSet.error(r.getStartLoc(),
                "cannot assign "+
                TagConstants.toString(ParseUtil.getUniverse(r))+
                " to "+
                TagConstants.toString(ParseUtil.getUniverse(l)));
	else if (!universeIsSubtypeOf(ParseUtil.getElementUniverse(r),
                                      ParseUtil.getElementUniverse(l)))
	    ErrorSet.error(r.getStartLoc(),
		"cannot assign "+
                TagConstants.toString(ParseUtil.getUniverse(r))+
                " "+
                TagConstants.toString(ParseUtil.getElementUniverse(r))+
                " to "+
                TagConstants.toString(ParseUtil.getUniverse(l))+
                " "+
                TagConstants.toString(ParseUtil.getElementUniverse(l)));
    }
    //alx-end

    /**
     * Tests if the cast of the universe modifiers <code>l</code>  
     * <code>r</code> is ever possible.
     *
     */
    //alx: dw test if cast of universe modifier of l and r is ever possible
    //TODO: check if the specs are enough
    public static void checkUniverseCastability(/*@ non_null @*/ ASTNode l, 
						/*@ non_null @*/ ASTNode r) {
    	if (!universeIsSubtypeOf(ParseUtil.getUniverse(r),
				 ParseUtil.getUniverse(l)) && 
            !universeIsSubtypeOf(ParseUtil.getUniverse(l),
				 ParseUtil.getUniverse(r)))
	    ErrorSet.error(r.getStartLoc(),
			   "A "+
			   TagConstants.toString(ParseUtil.getUniverse(r))+
			   " instance can never be of universe type "+
			   TagConstants.toString(ParseUtil.getUniverse(l)));
    	else if (!universeIsSubtypeOf(ParseUtil.getElementUniverse(r),
				      ParseUtil.getElementUniverse(l)) && 
		 !universeIsSubtypeOf(ParseUtil.getElementUniverse(l),
				      ParseUtil.getElementUniverse(r)))
	    ErrorSet.error(r.getStartLoc(),
			   "A "+
			   TagConstants.toString(ParseUtil.getUniverse(r))+
			   " "+
			   TagConstants.toString(ParseUtil.getElementUniverse(r))+
			   " instance can never be of universe type "+
			   TagConstants.toString(ParseUtil.getUniverse(l))+
			   " "+
			   TagConstants.toString(ParseUtil.getElementUniverse(l)));
    }
    //alx-end


    //alx: dw test if universe modifiers of sub is subtype of them of sup
    //TODO: chceck how this method is used, add specs
    /**
     * Tests if the universe modifier <code>sub</code> is a subtype of 
     * the universe modifier <code>sup</code>.
     *
     */
    public static boolean universeIsSubtypeOf(int sub, int sup){
	if (sub==TagConstants.NULLLIT || sub==0 || sup==0)
	    return true;
	if (sub==TagConstants.THISEXPR || sub==TagConstants.IMPL_PEER)
	    sub=TagConstants.PEER;
	if (sup==TagConstants.THISEXPR || sup==TagConstants.IMPL_PEER)
	    sup=TagConstants.PEER;
	return (sup==sub || sup==TagConstants.READONLY); 
    }
    //alx-end    

    //------------------------------------------------------------

    // note e must already have been prepped!
    //@ requires e != null && sig != null;
    //@ requires sig.state >= TypeSig.PREPPED;
    protected void checkTypeDeclElem(TypeDeclElem e) {

        Assert.notNull(sig);
        Assert.notFalse(sig.state >= TypeSig.PREPPED);
        TypeDecl d = sig.getTypeDecl();
        boolean specOnly = d.specOnly;

        switch (e.getTag()) {
	
            case TagConstants.FIELDDECL: {		
                FieldDecl fd = (FieldDecl)e;

                // Process ModifierPragmas
                Env rootEnv = Modifiers.isStatic(fd.modifiers) ? rootSEnv : rootIEnv;
                checkModifierPragmaVec(fd.pmodifiers, fd, rootEnv);
                checkTypeModifiers(rootEnv, fd.type);

		//alx: dw check the universe annotations
                if (useUniverses) 
		    checkUniverseForField(fd);

                // Resolve the initializer of a field decl
                if (fd.init != null) {
                    leftToRight = true;
                    allowedExceptions.removeAllElements();
                    Assert.notFalse(allowedExceptions.size() == 0);
                    fd.init = checkInit(rootEnv, fd.init, fd.type);
                    //alx: dw check the universe annotations
                    if (useUniverses) {
                    	//Array-initializers don't know their universe type, 
			//we have to take it form the field declaration
                    	if (ParseUtil.getUniverse(fd.init)==0 && 
			    ParseUtil.getElementUniverse(fd.init)!=0)
			    ParseUtil.setUniverse(fd.init, 
						  ParseUtil.getUniverse(fd));
                    	checkUniverseAssignability(fd, fd.init);
                    }
                    //alx-end
                }
                else if (Modifiers.isFinal(fd.modifiers) && 
			 Modifiers.isStatic(fd.modifiers) && 
			 Modifiers.isPublic(fd.modifiers) && !specOnly) {
                    
		    String msg = ("Variable " + fd.id
				  + " might not have been initialized");
                    ErrorSet.caution(fd.locId, msg); 
                }
                break;
            }
	
            case TagConstants.METHODDECL:
            case TagConstants.CONSTRUCTORDECL: {
                RoutineDecl rd = (RoutineDecl) e;
                Env rootEnv = Modifiers.isStatic(rd.modifiers)
                    ? rootSEnv : rootIEnv;
	  
		//alx: dw remember if we are in a static or pure context
                inStatic=Modifiers.isStatic(rd.modifiers);
                inPure=isPure(rd);
		//alx-end

                // First do method/constructor specific stuff
                if(rd instanceof MethodDecl) {
	    
                    MethodDecl md = (MethodDecl) e;

                    checkTypeModifiers(rootEnv, md.returnType);
	      
                    returnType = md.returnType;

                    //alx:dw check that return type is not rep in a 
                    //    static context and remember universe type of 
                    //    the return value
                    if (useUniverses) {
                            checkNoRepInStaticContext(md);
                            if (ParseUtil.getUniverse(md)!=0) {
                            universeReturnType = ParseUtil.getUniverse(md);
                            if (ParseUtil.getElementUniverse(md)!=0)
                                universeElementReturnType = 
                                    ParseUtil.getElementUniverse(md);
                            }
                    }
                    //alx-end

                    if (md.body != null && !specOnly) {
              
                        if(Modifiers.isAbstract(md.modifiers))
                            ErrorSet.error(md.loc, 
                                           "An abstract method cannot include a body");
                        if(Modifiers.isNative(md.modifiers))
                            ErrorSet.error(md.loc, 
                                           "A native method cannot include a body");
                    } else {
/* We allow any method to have no body -- DRCok
                        if(!Modifiers.isAbstract(md.modifiers)
                           && !Modifiers.isNative(md.modifiers) && !specOnly)
                            ErrorSet.error(md.loc, 
                                           "Method must include a body unless "
                                           +"it is declared abstract or native");
*/
                    }
                } else {
                    // Constructor
                    ConstructorDecl cd = (ConstructorDecl)rd;

                    //alx:dw remember if in constructor
                    inCtor=true;
                    //alx-end

                    // Was checked in parser
                    Assert.notFalse(!(d instanceof InterfaceDecl)); //@ nowarn Pre;

                    // Modifiers were checked when we prep'ed the constructed
                    returnType = Types.voidType;

                    // Check if we need to add an implicit constructor invocation
                    //@ assume !specOnly ==> cd.body != null;
                    if(!dontAddImplicitConstructorInvocations && !specOnly &&
                        cd.body != null && // FIXME - we've broken the assumption above by allowing spec files - need to fix that uniformly
                        !(cd.body.stmts.size() > 0
                          && cd.body.stmts.elementAt(0) instanceof 
                          ConstructorInvocation)) {
                        // no explicit constructor invocation
                        if(sig != Types.javaLangObject()) {
                            // add implicit constructor invocation

                            ExprVec args = ExprVec.make();
                            ConstructorInvocation ci 
                                = ConstructorInvocation.make(true, null, Location.NULL,
                                                             cd.body.locOpenBrace,
                                                             cd.body.locOpenBrace, args);
                            cd.body.stmts.insertElementAt(ci, 0);
                        }
                    }
                }

                // Now do stuff common to methods and constructors.

                leftToRight = false;
                enclosingLabels.removeAllElements();
          
                allowedExceptions.removeAllElements();
                for(int j=0; j<rd.raises.size(); j++) {
                    TypeName n = rd.raises.elementAt(j);
                    rootEnv.resolveType(sig,n);
                    checkTypeModifiers(rootEnv, n);
                    allowedExceptions.addElement(TypeSig.getSig(n));
                }

                Env env = new EnvForEnclosedScope(rootEnv);
                for(int j = 0; j < rd.args.size(); j++) {
                    FormalParaDecl formal = rd.args.elementAt(j);
                    PrepTypeDeclaration.inst.
                        checkModifiers(formal.modifiers, Modifiers.ACC_FINAL, 
                                       formal.getStartLoc(), "formal parameter");
                    checkModifierPragmaVec(formal.pmodifiers, formal, rootEnv);

                    env = new EnvForLocals(env, formal);
                    checkTypeModifiers(env, formal.type);
                    //alx: dw test universe types of arguments
                    if (useUniverses) {
                            int tag=ParseUtil.getUniverse(formal);
                            if (tag!=0) { //is not a primitive type
                            if (inCtor && tag==TagConstants.REP)
                                ErrorSet.error(formal.getStartLoc(),
                                   "rep not allowed in constructor signature");
                            else if (!inCtor && inPure) {
                                if (tag!=TagConstants.IMPL_PEER && 
                                    tag!=TagConstants.READONLY)
                                    ErrorSet.error(formal.getStartLoc(),
                                               "arguments of pure methods"+
                                               " must have readonly "+
                                               "universe type");
                                ParseUtil.setUniverse(formal,
                                                  TagConstants.READONLY);
                            } else if (inCtor && 
                                       inPure && 
                                       readonlyStdForPureCtor && 
                                       tag==TagConstants.IMPL_PEER) {
                                ParseUtil.setUniverse(formal,
                                                      TagConstants.READONLY);
                            } else {
                                checkNoRepInStaticContext(formal);
                            }
                            }
                    }
                    //alx-end
                }

                // Process ModifierPragmas
                checkModifierPragmaVec(rd.pmodifiers, rd, env);
          
                if (rd.body != null) {
                    checkStmt(env, rd.body);
                }

                break;
            }
        
            case TagConstants.INITBLOCK: {
                InitBlock si = (InitBlock) e;
                PrepTypeDeclaration.inst.
                    checkModifiers(si.modifiers, Modifiers.ACC_STATIC, 
                                   si.getStartLoc(), "initializer body");
                Env rootEnv = Modifiers.isStatic(si.modifiers) ? rootSEnv : rootIEnv;
                returnType = null;
                checkStmt(new EnvForEnclosedScope(rootEnv), si.block);
                //alx:dw remember if in static context
                if (useUniverses)
                    inStatic = Modifiers.isStatic(si.modifiers);
                //alx-end
                break;
            }
        
            case TagConstants.CLASSDECL:
            case TagConstants.INTERFACEDECL: {
                TypeSig.getSig((TypeDecl)e).typecheck();
                break;
            }

            default:
                if(e instanceof TypeDeclElemPragma)
                    checkTypeDeclElemPragma((TypeDeclElemPragma)e);
                else
                    Assert.fail("Switch fall-through (" + e.getTag() + ")");
        }
        //alx:dw reset for the next TypeDeclElem
        inPure=false;
        inCtor=false;
        inStatic=false;
        universeReturnType = 0;
        universeElementReturnType = 0;
        //alx-end
    }
    
    // === Typecheck statements : begin ===

    protected Env checkVarDeclStmt(Env e, LocalVarDecl s) {
        e.resolveType(sig, s.type);
        checkTypeModifiers(e, s.type);
        PrepTypeDeclaration.inst.checkModifiers(
            s.modifiers, 
            Modifiers.ACC_FINAL,
            s.locId, 
            "local variable");
        checkModifierPragmaVec(s.pmodifiers, s, e);

        Env newEnv = new EnvForLocals(e, s);
        if (s.init != null) {
            s.init = checkInit(newEnv, s.init, s.type);
            //alx: dw set the universe type for array inits and test it
            if (useUniverses) {
                    if (ParseUtil.getUniverse(s.init) == 0 
                        &&  ParseUtil.getElementUniverse(s.init)!=0) {
                        ParseUtil.setUniverse(s.init, 
                            ParseUtil.getUniverse(s));
                    }
                    checkUniverseAssignability(s, s.init);
            }
            //alx-end
        }
        //alx: dw test if the given universe modifier is allowed 
        //   for this context
        if (useUniverses) {
            checkNoRepInStaticContext(s);
        }
        //alx-end
        return newEnv;
    }
    
    protected Env checkClassDeclStmt(Env e, ClassDeclStmt s) {
        // TODO: Check this!
        Env newEnv = new EnvForLocalType(e, s.decl);
        TypeSig T = Types.makeTypeSig(s.decl.id.toString(), newEnv, s.decl);
        T.typecheck();
        return newEnv;
    }
    
    protected Env checkSwitchStmt(Env e, SwitchStmt s) {
        s.expr = checkExpr(e, s.expr);
        Env env = e;

        // Now do special handling of the following block with case stmts
        Type switchType = getType(s.expr);
        if (!Types.isIntegralType(switchType) || Types.isLongType(switchType)) {
            ErrorSet.error(
                s.expr.getStartLoc(),
                "The type of a switch expression must be char,"
                    + " byte, short, or int.");
            switchType = Types.intType;
        }

        // What case values encountered so far.
        java.util.Set switchValues = new HashSet();

        boolean defaultEncountered = false;
        enclosingLabels.addElement(s);

        for (int i = 0, sz = s.stmts.size(); i < sz; i++) {
            Stmt stmt = s.stmts.elementAt(i);

            if (stmt.getTag() == TagConstants.SWITCHLABEL) {
                SwitchLabel x = (SwitchLabel)stmt;
                if (x.expr != null) {
                    x.expr = checkExpr(env, x.expr);
                    Object val = ConstantExpr.eval(x.expr);
                    // System.out.println("At
                    // "+Location.toString(x.expr.getStartLoc()));

                    if (val == null)
                        ErrorSet.error(
                            x.loc,
                            "Non-constant value in switch label");
                    else if (!ConstantExpr.constantValueFitsIn(
                        val,
                        (PrimitiveType)switchType))
                        ErrorSet.error(x.loc, "Case label value (" + val
                            + ") not assignable to "
                            + "the switch expression type "
                            + Types.printName(switchType));
                    else {
                        // Check if it is a duplicate
                        // val may be Integer or Long, convert to Long for
                        // duplicate checking
                        Assert.notFalse(val instanceof Long //@ nowarn Pre;
                            || val instanceof Integer);
                        Long valLong = new Long(
                            ConstantExpr.getLongConstant(val));
                        if (switchValues.contains(valLong)) {
                            // Point to dup label - FIXME
                            ErrorSet.error(x.loc, "Duplicate case label " + val
                                + " in switch statement");
                        } else {
                            switchValues.add(valLong);
                        }
                    }
                } else {
                    // this is default
                    if (defaultEncountered)
                        // Point to dup label - FIXME
                        ErrorSet.error(
                            x.loc,
                            "Duplicate default label in switch statement");
                    else
                        defaultEncountered = true;
                }

            } else
                env = checkStmt(env, stmt);
        }

        enclosingLabels.pop();
        return e;
    }

    /**
     * Typecheck a statement in a given environment then return the environment
     * in effect for statements that follow the given statement.
     * <p>
     * (The returned environment will be the same as the one passed in unless
     * the statement is a declaration.)
     * </p>
     */
    //@ requires e != null && s != null;
    //@ requires !(e instanceof EnvForCU);
    //@ requires sig != null;
    //@ ensures \result != null;
    //@ ensures !(\result instanceof EnvForCU);
    protected Env checkStmt(Env e, Stmt s) {

        switch (s.getTag()) {
        // === Handle declarations ===
        case TagConstants.VARDECLSTMT:
            return checkVarDeclStmt(e, ((VarDeclStmt)s).decl);
        case TagConstants.CLASSDECLSTMT:
            return checkClassDeclStmt(e, (ClassDeclStmt)s);

        // === Handle `switch' and related statements ===
        // NOTE: The block following `switch' is also handled.
        case TagConstants.SWITCHLABEL:
            Assert.precondition("Switch label passed to checkStmt!");
        case TagConstants.SWITCHSTMT:
            return checkSwitchStmt(e, (SwitchStmt)s);

        // === Handle the rest of the statements ===
        case TagConstants.BREAKSTMT:
        case TagConstants.CONTINUESTMT: {
            BranchStmt bs = (BranchStmt)s;
            Stmt dest = null;
            int size = enclosingLabels.size();

            String expectedStmtKind = s.getTag() == TagConstants.BREAKSTMT ? "switch, while, do, or for"
                : "while, do or for";

            for (int i = size - 1; i >= 0 && dest == null; i--) {
                Stmt ati = enclosingLabels.elementAt(i);
                Stmt target = ati instanceof LabelStmt ? ((LabelStmt)ati).stmt
                    : ati;

                // continue target must be a loop statement
                // unlabelled break target must be a loop or switch
                // labelled break can be any statement

                boolean loopTarget = (target instanceof ForStmt)
                    || (target instanceof WhileStmt)
                    || (target instanceof DoStmt);

                boolean validTarget = loopTarget
                    || (s.getTag() == TagConstants.BREAKSTMT && (target instanceof SwitchStmt || bs.label != null));

                if (bs.label == null) {
                    if (validTarget)
                        dest = target;
                    else
                        continue;
                } else if (ati instanceof LabelStmt
                    && ((LabelStmt)ati).label == bs.label) {
                    if (!validTarget)
                        ErrorSet.caution(
                            bs.loc,
                            "Enclosing statement labelled '" + bs.label
                                + "' is not a " + expectedStmtKind
                                + " statement");
                    // just give a warning and continue, since javac accepts
                    // this
                    dest = target;
                }
                // else continue
            }

            if (dest == null) {
                ErrorSet.error(
                    bs.loc,
                    bs.label == null ? "No enclosing unlabelled "
                        + expectedStmtKind + " statement" : "No enclosing "
                        + expectedStmtKind + " statement labelled '" + bs.label
                        + "'");
            } else
                setBranchLabel(bs, dest);

            return e;
        }

        case TagConstants.SKIPSTMT:
            return e;

        case TagConstants.RETURNSTMT: {
            ReturnStmt rs = (ReturnStmt)s;

            if (rs.expr != null) rs.expr = checkExpr(e, rs.expr);

            if (returnType == null)
                ErrorSet.error(
                    rs.loc,
                    "Returns are not allowed in a static initializer");
            else {
                if (rs.expr != null) {
                    Type res = getType(rs.expr);

                    // alx: dw test universe of return stmt
                    if (useUniverses
                        && !universeIsSubtypeOf(
                            universeReturnType,
                            ParseUtil.getUniverse(rs.expr)))
                        ErrorSet.error(
                            rs.getStartLoc(),
                            "This routine must return "
                                + TagConstants.toString(universeReturnType)
                                + ", not "
                                + TagConstants.toString(ParseUtil.getUniverse(rs.expr)));
                    else if (useUniverses
                        && !universeIsSubtypeOf(
                            universeElementReturnType,
                            ParseUtil.getElementUniverse(rs.expr)))
                        ErrorSet.error(
                            rs.getStartLoc(),
                            "This routine must return "
                                + TagConstants.toString(universeReturnType)
                                + " "
                                + TagConstants.toString(universeElementReturnType)
                                + ", not "
                                + TagConstants.toString(ParseUtil.getUniverse(rs.expr))
                                + " "
                                + TagConstants.toString(ParseUtil.getElementUniverse(rs.expr)));
                    // alx-end

                    if (Types.isSameType(returnType, Types.voidType))
                        ErrorSet.error(
                            rs.loc,
                            "This routine is not expected to return a value");
                    else if (!assignmentConvertable(rs.expr, returnType)) {
                        if (!Types.isErrorType(res))
                            ErrorSet.error(rs.loc, "Cannot convert "
                                + Types.printName(res) + " to return type "
                                + Types.printName(returnType));
                    }
                } else {
                    if (!Types.isSameType(returnType, Types.voidType))
                        ErrorSet.error(
                            rs.loc,
                            "This routine must return a value");
                }
            }
            return e;
        }

        case TagConstants.THROWSTMT: {
            ThrowStmt t = (ThrowStmt)s;
            t.expr = checkExpr(e, t.expr);
            Type res = getType(t.expr);

            if (!Types.isSubclassOf(res, Types.javaLangThrowable())) {
                ErrorSet.error(t.loc, "Cannot throw values of type "
                    + Types.printName(res));
            } else {

                if (Types.isCheckedException(res)) {

                    // Must be caught by try or throws clause
                    for (int i = 0; i < allowedExceptions.size(); i++) {
                        if (Types.isSubclassOf(
                            res,
                            allowedExceptions.elementAt(i)))
                        // is ok
                            return e;
                    }
                    // Not caught
                    ErrorSet.error(
                        t.loc,
                        "Exception must be caught by an enclosing try "
                            + "or throws clause");
                }
            }
            return e;
        }

        case TagConstants.BLOCKSTMT:
            checkStmtVec(e, ((GenericBlockStmt)s).stmts);
            return e;

        case TagConstants.WHILESTMT: {
            WhileStmt w = (WhileStmt)s;
            w.expr = checkExpr(e, w.expr, Types.booleanType);
            enclosingLabels.addElement(w);
            checkStmt(e, w.stmt);
            enclosingLabels.pop();
            return e;
        }

        case TagConstants.DOSTMT: {
            DoStmt d = (DoStmt)s;
            d.expr = checkExpr(e, d.expr, Types.booleanType);
            enclosingLabels.addElement(d);
            checkStmt(e, d.stmt);
            enclosingLabels.pop();
            return e;
        }

        case TagConstants.IFSTMT: {
            IfStmt i = (IfStmt)s;
            i.expr = checkExpr(e, i.expr, Types.booleanType);
            checkStmt(e, i.thn);
            checkStmt(e, i.els);
            return e;
        }

        case TagConstants.SYNCHRONIZESTMT: {
            SynchronizeStmt ss = (SynchronizeStmt)s;
            ss.expr = checkExpr(e, ss.expr, Types.javaLangObject());
            checkStmt(e, ss.stmt);
            return e;
        }

        case TagConstants.EVALSTMT: {
            EvalStmt v = (EvalStmt)s;
            v.expr = checkExpr(e, v.expr);
            return e;
        }

        case TagConstants.LABELSTMT: {
            LabelStmt ls = (LabelStmt)s;
            for (int i = 0; i < enclosingLabels.size(); i++) {
                Stmt enclosing = enclosingLabels.elementAt(i);
                if (enclosing instanceof LabelStmt
                    && ((LabelStmt)enclosing).label == ls.label)
                    ErrorSet.error(ls.locId, "Label '" + ls.label
                        + "' already used in this scope");
                // FIXME - point to dup
            }

            enclosingLabels.addElement(ls);
            checkStmt(e, ls.stmt);
            enclosingLabels.pop();

            return e;
        }

        case TagConstants.TRYFINALLYSTMT: {
            TryFinallyStmt tf = (TryFinallyStmt)s;
            checkStmt(e, tf.tryClause);
            checkStmt(e, tf.finallyClause);
            return e;
        }

        case TagConstants.TRYCATCHSTMT: {
            TryCatchStmt tc = (TryCatchStmt)s;
            TypeSigVec newExceptions = allowedExceptions.copy();

            // add all catch clause exceptions to allowedExceptions
            for (int i = 0; i < tc.catchClauses.size(); i++) {
                CatchClause c = tc.catchClauses.elementAt(i);
                Type t = c.arg.type;
                e.resolveType(sig, t);
                checkTypeModifiers(e, t);
                if (!Types.isSubclassOf(t, Types.javaLangThrowable()))
                    ErrorSet.error(
                        c.loc,
                        "Declared type of a catch formal parameter "
                            + "must be a subclass of java.lang.Throwable");
                else {
                    if (t instanceof TypeName) {
                        t = TypeSig.getSig((TypeName)t);
                    }
                    newExceptions.addElement((TypeSig)t);
                }
                PrepTypeDeclaration.inst.checkModifiers(
                    c.arg.modifiers,
                    Modifiers.ACC_FINAL,
                    c.arg.getStartLoc(),
                    "formal parameter");

                EnvForLocals subenv = new EnvForLocals(e, c.arg, false);
                checkStmt(subenv, c.body);
            }

            TypeSigVec oldExceptions = allowedExceptions;
            allowedExceptions = newExceptions;
            checkStmt(e, tc.tryClause);
            allowedExceptions = oldExceptions;

            return e;
        }

        case TagConstants.FORSTMT: {
            ForStmt f = (ForStmt)s;
            Env se = checkStmtVec(e, f.forInit);
            checkForLoopAfterInit(se, f);
            return e;
        }

        case TagConstants.CONSTRUCTORINVOCATION: {
            ConstructorInvocation ci = (ConstructorInvocation)s;

            TypeSig calleeSig = ci.superCall ? TypeSig.getSig(((ClassDecl)sig.getTypeDecl()) //@nowarn Cast, NonNull;
                .superClass)
                : sig;

            /*
             * Everything before the constructor invocation call occurs is
             * considered to be in a static context:
             */
            Env sEnv = e.asStaticContext();

            Type[] argTypes = checkExprVec(sEnv, ci.args);

            /*
             * Handle type checking/inference for enclosing instance ptr:
             */

            // Get the type of calleeSig's enclosing instance or null if none
            // exists:
            TypeSig enclosingInstanceType = calleeSig.getEnv(false).getEnclosingInstance();

            // If calleeSig has an enclosing instance and the user didn't supply
            // a value for it, try to infer one:
            if (ci.enclosingInstance == null && enclosingInstanceType != null) {
                ci.locDot = ci.getStartLoc();
                ci.enclosingInstance = sEnv.lookupEnclosingInstance(
                    enclosingInstanceType,
                    ci.getStartLoc());
            }

            if (ci.enclosingInstance != null) {
                if (enclosingInstanceType != null)
                    ci.enclosingInstance = checkExpr(
                        sEnv,
                        ci.enclosingInstance,
                        enclosingInstanceType);
                else
                    ErrorSet.error(
                        ci.enclosingInstance.getStartLoc(),
                        "An enclosing instance may be provided only "
                            + "when the superclass has an enclosing instance");
            } else if (enclosingInstanceType != null) {
                /*
                 * Compute appropriate error message details:
                 */
                String details;
                if (ci.locKeyword == sig.getTypeDecl().locOpenBrace) {
                    // inferred constructor with an inferred super inside it:
                    details = "cannot create a default constructor for class "
                        + sig + ".";
                } else if (ci.locKeyword == ci.locOpenParen) {
                    // inferred super(...) in an explicit constructor:
                    details = "cannot create a default superclass constructor."
                        + "  (superclass is " + calleeSig + ")";
                } else {
                    // explicit super(...) case...
                    details = "an explicit one must be"
                        + " provided when creating inner class " + calleeSig
                        + ".";
                }

                ErrorSet.error(
                    ci.getStartLoc(),
                    "No enclosing instance of class " + enclosingInstanceType
                        + " is in scope; " + details);
            }

            try {
                ConstructorDecl cd = calleeSig.lookupConstructor(argTypes, sig);
                Assert.notNull(cd);
                ci.decl = cd;
            } catch (LookupException ex) {
                // Don't print an error if superclass is an interface (an
                // error has already been reported in this case)
                if (!ci.superCall
                    || calleeSig.getTypeDecl().getTag() == TagConstants.CLASSDECL)
                    reportLookupException(
                        ex,
                        "constructor " + calleeSig.getTypeDecl().id
                            + Types.printName(argTypes),
                        Types.printName(calleeSig),
                        ci.locKeyword);
            }

            // Rest of constructor body is in the normal non-static context:
            return e;
        }

        case TagConstants.ASSERTSTMT: {
            AssertStmt assertStmt = (AssertStmt)s;
            if (assertStmt.label != null)
                assertStmt.label = checkExpr(e, assertStmt.label);
            assertStmt.pred = checkExpr(e, assertStmt.pred, Types.booleanType);

            // Turn a Java assert into a conditional throw and attach it to the
            // AssertStmt.

            // assert Predicate ;
            // ==>
            // if (! Predicate)
            // throw new java.lang.AssertionError();
            // or
            // assert Predicate : LabelExpr ;
            // ==>
            // if (! Predicate)
            // throw new java.lang.AssertionError(LabelExpr);
            int startLoc = assertStmt.getStartLoc();
            int endLoc = assertStmt.getEndLoc();
            Assert.notFalse(startLoc != Location.NULL);
            UnaryExpr negatedPredicate = UnaryExpr.make(
                TagConstants.NOT,
                assertStmt.pred,
                startLoc);
            ParenExpr parenthesizedNegatedPredicate = ParenExpr.make(
                negatedPredicate,
                startLoc,
                endLoc);
            Name javaLangAssertionErrorName = Name.make(
                "java.lang.AssertionError",
                startLoc);
            TypeName javaLangAssertionTypeName = TypeName.make(javaLangAssertionErrorName);
            ExprVec constructorArgs = null;
            if (assertStmt.label != null) {
                Expr[] constructorArgsAsArray = { assertStmt.label };
                constructorArgs = ExprVec.make(constructorArgsAsArray);
            } else {
                constructorArgs = ExprVec.make();
            }
            NewInstanceExpr newAssertionError = NewInstanceExpr.make(
                null,
                Location.NULL,
                javaLangAssertionTypeName,
                constructorArgs,
                null,
                startLoc,
                endLoc);
            ThrowStmt throwAssertionError = ThrowStmt.make(
                newAssertionError,
                startLoc);
            Stmt elseSkip = SkipStmt.make(startLoc);
            IfStmt ifStmt = IfStmt.make(
                parenthesizedNegatedPredicate,
                throwAssertionError,
                elseSkip,
                startLoc);
            // Now typecheck the generated IfStmt so that it is fully resolved.
            ifStmt.expr = checkExpr(e, ifStmt.expr, Types.booleanType);
            checkStmt(e, ifStmt.thn);
            checkStmt(e, ifStmt.els);
            // Reattach this translated form to the AST node.
            assertStmt.ifStmt = ifStmt;

            return e;
        }

        default:
            if (s instanceof StmtPragma)
                checkStmtPragma(e, (StmtPragma)s);
            else {
                Assert.fail("Switch fall-through (" + s.getTag() + " "
                    + TagConstants.toString(s.getTag()) + " "
                    + Location.toString(s.getStartLoc()) + ")");
            }
        }
        return e; // dummy
    }
    // === Typecheck statements : end ===
    

    //@ requires !(se instanceof EnvForCU);
    //@ requires sig != null;
    protected void checkForLoopAfterInit(/*@non_null*/ Env se,
                                         /*@non_null*/ ForStmt f) {
        f.test = checkExpr(se, f.test);
        for (int i = 0; i < f.forUpdate.size(); i++) {
            Expr ex = checkExpr(se, f.forUpdate.elementAt(i));
            f.forUpdate.setElementAt(ex, i);
        }
        enclosingLabels.addElement(f);
        checkStmt(se, f.body);
        enclosingLabels.pop();
    }

    //@ requires env != null && v != null;
    //@ requires !(env instanceof EnvForCU);
    //@ requires sig != null;
    //@ ensures \result != null;
    //@ ensures !(\result instanceof EnvForCU);
    protected Env checkStmtVec(Env env, StmtVec v) {
        for(int i = 0, sz = v.size(); i < sz; i++) {
            Stmt stmt = v.elementAt(i);
            
            env = checkStmt(env, stmt);
        }

        return env;
    }

  
    //@ requires env != null && ev != null ;
    //@ requires !(env instanceof EnvForCU);
    //@ requires sig != null;
    //@ ensures \nonnullelements(\result);
    protected Type[] checkExprVec(Env env, ExprVec ev) {

        Type[] r = new Type[ ev.size() ];

        for(int i = 0; i < ev.size(); i++) {
            Expr expr = ev.elementAt(i);
            Expr newExpr = checkExpr(env, expr);
            ev.setElementAt(newExpr, i);
            r[i] = getType(newExpr);
        }

        return r;
    }


    //@ requires sig != null;
    //@ requires !(env instanceof EnvForCU);
    //@ requires env != null && x != null && expectedResult != null;
    //@ ensures \result != null;
    //@ ensures (x instanceof ArrayInit) ==> \result instanceof ArrayInit;
    protected VarInit checkInit(Env env, VarInit x, Type expectedResult) {
        if (x instanceof ArrayInit) {
            ArrayInit ai = (ArrayInit)x;

            if(expectedResult instanceof ArrayType) {
                //alx: dw remember least upper bound for universes
                    int bound=TagConstants.NULLLIT;
                //alx-end

                Type elemType = ((ArrayType)expectedResult).elemType;
                for(int i=0; i< ai.elems.size(); i++) {
                    VarInit disamb = checkInit(env, ai.elems.elementAt(i), elemType);
                    ai.elems.setElementAt(disamb, i);
                    //alx: dw TODO: explain that addition
                    bound=leastUpperUniverseBound(bound,
                                           ParseUtil.getUniverse(disamb));
                    //alx-end
                }
                //alx: dw save bound as universe for the elements
                ParseUtil.setElementUniverse(ai,bound);
                //alx-end

                setType(ai, expectedResult);
            } else {
                ErrorSet.error(ai.locOpenBrace, 
                               "Unexpected array value in initializer");
            }
            return x;

        } else {
            //@ assume \typeof(x) <: \type(Expr);
            return checkExpr(env, (Expr) x, expectedResult);
        }
    }

    //@ requires env != null && e != null;
    //@ requires !(env instanceof EnvForCU);
    //@ requires sig != null;
    //@ ensures \result != null;
    protected Expr checkDesignator(Env env, Expr e) {
        Expr result = checkExpr(env, e);
        if (result.getTag() == TagConstants.FIELDACCESS) {
            FieldAccess fa = (FieldAccess)result;
            if (fa.decl == Types.lengthFieldDecl) {
                ErrorSet.error(fa.locId, "Invalid designator");
            }
            // Note, if it weren't for some Java 1.1 features, one could also
            // rule out the case:
            //   fa.decl != null && Modifiers.isFinal(fa.decl.modifiers)
            // here with a message "Final fields cannot be used as designators",
            // but Java 1.1 allows assignments to final fields in some contexts.
            // A more elaborate test could perhaps be used, though.
        }
        return result;
    }

    /**
     * Typechecks expressions.
     * @param env  The current environment.
     * @param expr The expression to be checked.
     * @param t    The expected type.
     * @return     The checked expression.
     */
    //@ requires !(env instanceof EnvForCU);
    //@ requires sig != null;
    //@ ensures \result == expr;
    protected Expr checkExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ Expr expr, 
        /*@ non_null */ Type t
    ) {
        Expr ne = checkExpr(env,expr);
        checkType(ne, t);
        return ne;
    }
    
    /**
     * Typechecks expressions. Work is delegated to specialized methods.
     * This way the clients that inherit from this class have a finer
     * grained control by overriding methods.
     * 
     * @param env The current environment.
     * @param x   The expression to typecheck.
     * @return    The checked expression. It has the type set.
     */
    //@ requires !(env instanceof EnvForCU);
    //@ requires sig != null;
    //@ ensures \result == x;
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkExpr(/*@ non_null */ Env env, /*@ non_null */ Expr x) {
        // System.out.println("Checking: "+Location.toString(x.getStartLoc()));

        if (getTypeOrNull(x) != null) {
            // already checked
            return x;
        }

        // Set default result type to error.
        setType(x, Types.errorType);

        switch (x.getTag()) {
        case TagConstants.THISEXPR:
            return checkThisExpr(env, (ThisExpr)x);

        // The environment is not needed to check literals.
        case TagConstants.STRINGLIT:
            return checkStringLitExpr(x);
        case TagConstants.CHARLIT:
            return checkCharLitExpr(x);
        case TagConstants.BOOLEANLIT:
            return checkBooleanLitExpr(x);
        case TagConstants.FLOATLIT:
            return checkFloatLitExpr(x);
        case TagConstants.DOUBLELIT:
            return checkDoubleLitExpr(x);
        case TagConstants.INTLIT: 
            return checkIntLitExpr(x);
        case TagConstants.LONGLIT:
            return checkLongLitExpr(x);
        case TagConstants.NULLLIT: 
            return checkNullLitExpr(x);

        case TagConstants.ARRAYREFEXPR:
            return checkArrayRefExpr(env, (ArrayRefExpr)x);
        case TagConstants.NEWINSTANCEEXPR:
            return checkNewInstanceExpr(env, (NewInstanceExpr)x);
        case TagConstants.NEWARRAYEXPR:
            return checkNewArrayExpr(env, (NewArrayExpr)x);
        case TagConstants.CONDEXPR:
            return checkCondExpr(env, (CondExpr)x);
        case TagConstants.INSTANCEOFEXPR:
            return checkInstanceOfExpr(env, (InstanceOfExpr)x);
        case TagConstants.CASTEXPR:
            return checkCastExpr(env, (CastExpr)x);

        case TagConstants.CLASSLITERAL:
            return checkClassLiteralExpr(env, (ClassLiteral)x);

        case TagConstants.OR: case TagConstants.AND:
        case TagConstants.BITOR: case TagConstants.BITXOR:
        case TagConstants.BITAND: case TagConstants.NE:
        case TagConstants.EQ: case TagConstants.GE:
        case TagConstants.GT: case TagConstants.LE:
        case TagConstants.LT: case TagConstants.LSHIFT:
        case TagConstants.RSHIFT: case TagConstants.URSHIFT:
        case TagConstants.ADD: case TagConstants.SUB:
        case TagConstants.DIV: case TagConstants.MOD:
        case TagConstants.STAR:
            return checkBinaryExpr(env, (BinaryExpr)x);
  
        case TagConstants.ASSIGN: case TagConstants.ASGMUL:
        case TagConstants.ASGDIV: case TagConstants.ASGREM:
        case TagConstants.ASGADD: case TagConstants.ASGSUB:
        case TagConstants.ASGLSHIFT: case TagConstants.ASGRSHIFT:
        case TagConstants.ASGURSHIFT: case TagConstants.ASGBITAND:
        case TagConstants.ASGBITOR: case TagConstants.ASGBITXOR: 
            return checkAssignmentExpr(env, (BinaryExpr)x);

        case TagConstants.UNARYADD: 
        case TagConstants.UNARYSUB:
            return checkSignExpr(env, (UnaryExpr)x);

        case TagConstants.BITNOT:
            return checkBitnotExpr(env, (UnaryExpr)x);

        case TagConstants.INC: case TagConstants.DEC: 
        case TagConstants.POSTFIXINC: case TagConstants.POSTFIXDEC:
            return checkIncDecExpr(env, (UnaryExpr)x);
  
        case TagConstants.NOT:
            return checkNotExpr(env, (UnaryExpr)x);
  
        case TagConstants.PARENEXPR:
            return checkParenExpr(env, (ParenExpr)x);

        case TagConstants.AMBIGUOUSVARIABLEACCESS:
            return checkAmbiguousVariableAccessExpr(env, (AmbiguousVariableAccess)x);
        case TagConstants.FIELDACCESS:
            return checkFieldAccessExpr(env, (FieldAccess)x);
        case TagConstants.AMBIGUOUSMETHODINVOCATION:
            return checkAmbiguousMethodInvocationExpr(env, (AmbiguousMethodInvocation)x);
        case TagConstants.METHODINVOCATION:
            return checkMethodInvocationExpr(env, (MethodInvocation)x);

        case TagConstants.VARIABLEACCESS:
            // TODO: No environment needed? (rgrig)
            return checkVariableAccessExpr((VariableAccess)x); 

        default:
            System.out.println("FAIL " + x);
            System.out.println(" AT " + Location.toString(x.getStartLoc())); 
            Assert.fail(
                "Switch fall-through (" + TagConstants.toString(x.getTag()) + ")"
            );
            //@ assert false;
            return null; // dummy
        }
    }

    /**
     * Typecheck variable access. 
     * @param lva The variable access.
     * @return    The variable access.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkVariableAccessExpr(/*@ non_null */ VariableAccess lva) {
        setType(lva, lva.decl.type);
        Assert.notNull(getType(lva));

        //alx: dw take the universes of the declaration
        if (useUniverses) {
            copyUniverses(lva, lva.decl);
        }
        //alx-end

        // Front-end VariableAccess's never point to fields.
        Assert.notFalse(!(lva.decl instanceof FieldDecl)); //@ nowarn Pre;

        // Make sure only access final variables from enclosing instances.
        if (Env.whereDeclared(lva.decl) != sig && 
            !Modifiers.isFinal(lva.decl.modifiers)) {
            ErrorSet.error(
                lva.loc,
                "Attempt to use a non-final variable from a different " +
                "method.  From enclosing blocks, only final local variables " +
                "are available."
            );
        }

        return lva;
    }

    /**
     * Typecheck method invocation.
     * @param env The current environment.
     * @param mi  The method invocation.
     * @return    The checked method invocation.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected MethodInvocation checkMethodInvocationExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ MethodInvocation mi
    ) {
        Type t = checkObjectDesignator(env, mi.od);
        Type[] argTypes = checkExprVec(env, mi.args);

        // TODO: Shouldn't an error be reported for [t == null]? (rgrig)
        if (t != null) {
            try {
                mi.decl = Types.lookupMethod(t, mi.id, argTypes, sig);
                setType(mi, mi.decl.returnType);

                //alx: dw determine or set default universe modifier
                if (useUniverses) {
                    determineUniverseForMethodInvocation(mi);
                }
                // alx-end

                // When [x.y] is correctly used, x being a class (type) implies
                // that y is a static method; otherwise something is wrong. 
                if (mi.od instanceof TypeObjectDesignator &&
                    !Modifiers.isStatic(mi.decl.modifiers)) {
                    // Figure out what exactly is wrong.

                    // Is mi.decl an instance method of the same class as
                    // mi is part of?
                    boolean thisMethod = false;
                    if (mi.decl.parent != null) {
                        thisMethod = 
                            env.getEnclosingClass().isSubtypeOf(TypeSig.getSig(mi.decl.parent));
                    }

                    if (thisMethod ||
                        ((TypeObjectDesignator)mi.od).type instanceof TypeName) {
                        ErrorSet.error(
                            mi.locId, 
                            "An instance method may be invoked" +
                                " only via an object and/or from a non-static" +
                                " context or an inner class enclosed by a type" +
                                " possessing that method."
                        );
                    } else {
                        ErrorSet.error(
                            mi.locId,
                            "The instance methods of type " +
                                ((TypeObjectDesignator)mi.od).type +
                                " may not be invoked from type " + sig
                        );
                    }
                }
            } catch(LookupException ex) {
                if (!Types.isErrorType(t)) {
                    reportLookupException(
                        ex, 
                        "method " + mi.id + Types.printName(argTypes), 
                        Types.printName(t), 
                        mi.locId
                    );
                }
                setType(mi, Types.errorType);
            }
        }
        return mi;
    }

    /**
     * Disambiguates an ambiguous method invocation and then typechecks it.
     * @param env The current environment.
     * @param ami The ambiguous method invocation.
     * @return    The checked ambiguous invocation.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkAmbiguousMethodInvocationExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ AmbiguousMethodInvocation ami
    ) {
        MethodInvocation resolved = env.disambiguateMethodName(ami);
        Assert.notFalse(
            resolved != null, 
            "Disambiguation failure should result in a fatal error."
        );
        return checkExpr(env, resolved);
    }

    /**
     * Typechecks a field access (e.g., <tt>this.x</tt>).
     * @param env The current environment.
     * @param fa  The field access.
     * @return    The checked field access.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected FieldAccess checkFieldAccessExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ FieldAccess fa
    ) {
        Type t = checkObjectDesignator(env, fa.od);
        if (t != null) {
            try {
                fa.decl = Types.lookupField(t, fa.id, sig);
                setType(fa, fa.decl.type);

                // TODO: The following code duplicates a lot of the code in checkMethodInvocation.
                // Given [x.y]. It should hold that if x is a class then y is static.
                if (fa.od instanceof TypeObjectDesignator &&
                    !Modifiers.isStatic(fa.decl.modifiers)) {
                    // Determine what error should be produced.
                    
                    // Is fa.decl an instance field of the same class as
                    // fa is part of?
                    boolean thisField = false;
                    if (fa.decl.parent != null) {
                        thisField = 
                            env.getEnclosingClass().isSubtypeOf(TypeSig.getSig(fa.decl.parent));
                    }
                    if (thisField ||
                        ((TypeObjectDesignator)fa.od).type instanceof TypeName) {
                        ErrorSet.error(
                            fa.locId,
                            "An instance field may be accessed only via " +
                                "an object and/or from a non-static" +
                                " context or an inner class enclosed" +
                                " by a type possessing that field."
                        );
                    } else {
                        // FIXME - point to declaration
                        // _Also_ point to [fa.decl.locId]? (rgrig)
                        ErrorSet.error(
                            fa.locId,
                            "The instance fields of type " +
                                ((TypeObjectDesignator)fa.od).type +
                                " may not be accessed from type " + sig
                        );
                        // FIXME = point to declaration
                    }
                }
                //alx: dw determine or set universe type
                if (useUniverses) {
                    determineUniverseForFieldAccess(fa);
                }
                //alx-end
            } catch(LookupException ex) {
                if (!Types.isErrorType(t)) {
                    reportLookupException(ex, "field", Types.printName(t), fa.locId);
                }
                setType(fa, Types.errorType);
            }
        } 
        // alx: dw if no target, just copy universe modifier of declaration. 
        // does this ever happen???
        else if (useUniverses) {
            copyUniverses(fa, fa.decl);
        }
        //alx-end
        return fa;
    }

    
    /**
     * Disambiguates an ambiguous variable access and then typechecks
     * (e.g., <tt>x</tt>).
     * @param env The current environment.
     * @param av  The variable access.
     * @return    The disambiguated and checked variable access.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkAmbiguousVariableAccessExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ AmbiguousVariableAccess av
    ) {
        Assert.notNull(av.name);
        Expr resolved = env.disambiguateExprName(av.name);
        if(resolved == null) {
            if (av.name.size() == 1) {
                ErrorSet.error(
                    av.getStartLoc(),
                    "Undefined variable '" + av.name.printName() + "'"
                );
            } else {
                ErrorSet.error(
                    av.getStartLoc(),
                    "Cannot resolve variable access '" +
                    av.name.printName() + "'"
                );
            }
            setType(av, Types.errorType);
            return av;
        }
        Assert.notFalse(resolved.getTag() != TagConstants.AMBIGUOUSVARIABLEACCESS); 
        return checkExpr(env, resolved);
    }

    
    /**
     * Typechecks paranthesis.
     * @param env The current environment.
     * @param pe  The paranthesis expression.
     * @return    The checked paranthesis expression.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkParenExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ ParenExpr pe
    ) {
        pe.expr = checkExpr(env, pe.expr);
        setType(pe, getType(pe.expr));
        //alx: dw copy the universes to the paren-expr
        if (useUniverses) {
            copyUniverses(pe,pe.expr);
        }
        //alx-end
        return pe;
    }

    
    /**
     * Typecheck boolean not (e.g., <tt>!x</tt>). 
     * The argument must be boolean and the result is a boolean.
     * @param env The current environment.
     * @param ue  The unary expression containing the boolean not operation.
     * @return The typechecked not expression.
     */
    //@ requires ue.getTag() == TagConstants.NOT;
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkNotExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ UnaryExpr ue
    ) {
        // Argument must be boolean, result is boolean
        ue.expr = checkExpr(env, ue.expr, Types.booleanType);
        setType(ue, Types.booleanType);
        return ue;
    }

    
    /**
     * Typecheck unary increment and decrement operators (e.g., <tt>++x</tt>). 
     * @param env The current environment.
     * @param ue  The unary expression representing the inc(dec)rement.
     * @return The checked expression.
     */
    /*@ requires ue.getTag() == TagConstants.INC || 
                 ue.getTag() == TagConstants.DEC ||
                 ue.getTag() == TagConstants.POSTFIXINC ||
                 ue.getTag() == TagConstants.POSTFIXDEC;
        ensures getTypeOrNull(\result) != null; */
    protected Expr checkIncDecExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ UnaryExpr ue
    ) {
        ue.expr = checkDesignator(env, ue.expr);
        if (ue.expr instanceof VariableAccess) {
            GenericVarDecl v = ((VariableAccess)ue.expr).decl;
            if (Modifiers.isFinal(v.modifiers)) {
                ErrorSet.caution(
                    ue.expr.getStartLoc(),
                    "May not assign to a final variable",
                    v.getStartLoc()
                );
            }
        } else if (ue.expr instanceof FieldAccess) {
            GenericVarDecl v = ((FieldAccess)ue.expr).decl;
            // v is null if there was an error such as a field
            // name that does not exist
            if (v == Types.lengthFieldDecl) { 
                ErrorSet.error(
                    ue.expr.getStartLoc(),
                    "May not assign to array's length field"
                );
            }
            else if (v != null && Modifiers.isFinal(v.modifiers)) {
                ErrorSet.caution(
                    ue.expr.getStartLoc(),
                    "May not assign to a final field",
                    v.getStartLoc()
                );
            }
        }

        // FIXME - need checks of more complicated expressions

        // Argument must be primitive numeric variable type
        if (checkNumericType(ue.expr)) {
            if(!isVariable(ue.expr))
                ErrorSet.error(ue.locOp,
                        "Argument of increment/decrement operation "
                        +"is not a location");
            // Result is of same type
            setType(ue, getType(ue.expr));
        }
        return ue;
    }

    
    /**
     * Typecheck bitwise negation (e.g., <tt>~x</tt>).
     * The argument must be a primitive numeric type; the result
     * has the same type as the argument. 
     * @param env The current environment.
     * @param ue  The unary expression representing bitwise negation.
     * @return The checked expression.
     */
    //@ requires ue.getTag() == TagConstants.BITNOT;
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkBitnotExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ UnaryExpr ue
    ) {
        ue.expr = checkExpr(env, ue.expr);
        if(checkIntegralType(ue.expr)) {
            setType(ue, Types.unaryPromote(getType(ue.expr)));
        } 
        return ue;
    }

    /**
     * Typecheck sign (e.g. <tt>+x</tt> or <tt>-x</tt>).
     * The argument must be a primitive numeric type; the result
     * has the same type as the argument.
     * @param env The current environment;
     * @param ue  The unary expression representing the sign.
     * @return The checked expression.
     */
    /*@ requires ue.getTag() == TagConstants.UNARYADD ||
                 ue.getTag() == TagConstants.UNARYSUB;
        ensures getTypeOrNull(\result) != null; */
    protected Expr checkSignExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ UnaryExpr ue
    ) {
        ue.expr = checkExpr(env, ue.expr);
        Type t = getType(ue.expr);
        if(checkNumericType(ue.expr)) {
            setType(ue, Types.unaryPromote(getType(ue.expr)));
        }
        return ue;
    }

    /**
     * Typecheck assignment-like expressions (e.g., <tt>a += 2</tt>).
     * @param env The current environment.
     * @param be  The binary expression representing the assignment.
     * @return The checked expression.
     */
    /*@ requires be.getTag() == TagConstants.ASSIGN ||
                 be.getTag() == TagConstants.ASGMUL ||
                 be.getTag() == TagConstants.ASGDIV ||
                 be.getTag() == TagConstants.ASGREM ||
                 be.getTag() == TagConstants.ASGADD ||
                 be.getTag() == TagConstants.ASGSUB ||
                 be.getTag() == TagConstants.ASGLSHIFT ||
                 be.getTag() == TagConstants.ASGRSHIFT ||
                 be.getTag() == TagConstants.ASGURSHIFT ||
                 be.getTag() == TagConstants.ASGBITAND ||
                 be.getTag() == TagConstants.ASGBITOR ||
                 be.getTag() == TagConstants.ASGBITXOR;
        ensures getTypeOrNull(\result) != null; */
    protected Expr checkAssignmentExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ BinaryExpr be
    ) {
        be.left = checkDesignator(env, be.left);
        be.right = checkExpr(env, be.right);
        Type t = checkBinaryExpr(be.op, be.left, be.right, be.locOp);
        setType(be, t);
        return be;
    }

    /**
     * Typecheck simple binary expressions (e.g., <tt>x + y</tt>).
     * @param env The current environment.
     * @param be  The simple binary expression.
     * @return    The checked expression.
     */
    // TODO: It's ugly to have two methods with the same name. (rgrig)
    /*@ requires be.getTag() == TagConstants.OR ||
                 be.getTag() == TagConstants.AND ||
                 be.getTag() == TagConstants.BITOR ||
                 be.getTag() == TagConstants.BITXOR ||
                 be.getTag() == TagConstants.BITAND ||
                 be.getTag() == TagConstants.NE ||
                 be.getTag() == TagConstants.EQ ||
                 be.getTag() == TagConstants.GE ||
                 be.getTag() == TagConstants.GT ||
                 be.getTag() == TagConstants.LE ||
                 be.getTag() == TagConstants.LT ||
                 be.getTag() == TagConstants.LSHIFT ||
                 be.getTag() == TagConstants.RSHIFT ||
                 be.getTag() == TagConstants.URSHIFT ||
                 be.getTag() == TagConstants.ADD ||
                 be.getTag() == TagConstants.SUB ||
                 be.getTag() == TagConstants.DIV ||
                 be.getTag() == TagConstants.MOD ||
                 be.getTag() == TagConstants.STAR;
        ensures getTypeOrNull(\result) != null; */
    protected Expr checkBinaryExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ BinaryExpr be
    ) {
        be.left = checkExpr(env, be.left);
        be.right = checkExpr(env, be.right);
        Type t = checkBinaryExpr(be.op, be.left, be.right, be.locOp);
        //alx: dw set universe for the binary expression, if any
        if (useUniverses) {
            copyUniverses(be,be.left);
        }
        // alx-end
        setType(be, t);
        return be;
    }

    /**
     * Typechecks a class literal (e.g., <tt>X</tt> which might appear
     * as the rhs of <tt>instanceof</tt>).
     * @param env The current environment.
     * @param cl  The class literal to check.
     * @return    The checked expression.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkClassLiteralExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ ClassLiteral cl
    ) {
        env.resolveType(sig,cl.type);
        checkTypeModifiers(env, cl.type); 
        setType(cl, Types.javaLangClass());
        //alx: dw is a field => set to IMPL_PEER
        ParseUtil.setUniverse(cl,TagConstants.IMPL_PEER);
        //alx-end
        return cl;
    }

    /**
     * Typecheck a (dynamic) cast.
     * @param env The current environment.
     * @param ce  The cast to be checked.
     * @return    The checked expression.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected CastExpr checkCastExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ CastExpr ce
    ) {
        ce.expr = checkExpr(env, ce.expr);
        Type exprType = getType(ce.expr);
        env.resolveType(sig, ce.type);
        checkTypeModifiers(env, ce.type); 
 
        if(!Types.isCastable(exprType, ce.type)) {
            ErrorSet.error(
                ce.locOpenParen, 
                "Bad cast from "+Types.printName(exprType) + " to " +
                    Types.printName(ce.type)
            );
        }
        //alx: dw check castability and 'no rep in static context'
        if (useUniverses) {
            checkUniverseCastability(ce, ce.expr);
            checkNoRepInStaticContext(ce);
        }
        //alx-end
        setType(ce, ce.type);
        return ce;
    }

    /**
     * Typechecks a dynamic type probe (e.g., <tt>x instanceof X</tt>).
     * @param env The current environment.
     * @param ie  The <tt>instance of</tt> expression.
     * @return    The checked expression.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkInstanceOfExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ InstanceOfExpr ie
    ) {
        ie.expr = checkExpr(env, ie.expr);
        Type exprType = getType(ie.expr);
        env.resolveType(sig, ie.type);
        checkTypeModifiers(env, ie.type);
        if(!Types.isReferenceType(ie.type)) {
            ErrorSet.error(ie.loc, "Non-reference type in instanceof operation");
        }
        else if(!Types.isCastable(exprType, ie.type)) {
            ErrorSet.error(
                ie.loc, 
                "Invalid instanceof operation: " +
                    "A value of type " + Types.printName(exprType) +
                    " can never be an instance of " +
                    Types.printName(ie.type)
            );
        }
        //alx: dw check castability for instanceof-expr
        if (useUniverses) {
            checkUniverseCastability(ie,ie.expr);
            checkNoRepInStaticContext(ie);
        }
        //alx-end

        setType(ie, Types.booleanType);
        return ie;
    }

    /**
     * Typechecks a conditional expression (e..g, <tt>a == 0 ? x : 0</tt>).
     * @param env The current environment.
     * @param ce  The expression to check.
     * @return    The checked expression.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkCondExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ CondExpr ce
    ) {
        ce.test = checkExpr(env, ce.test, Types.booleanType);
        ce.thn = checkExpr(env, ce.thn);
        ce.els = checkExpr(env, ce.els);

        // The two branches should have compatible types (i.e., at least one
        // should be castable to the other).
        Type res = tryCondExprMatch(ce.thn, ce.els);
        if (res != null) {
            setType(ce, res);
        } else {
            ErrorSet.error(
                ce.locQuestion,
                "Incompatible arguments to the ?: operator"
            );
        }
        
        //alx: dw determine universe for 'b? (peer T) x: (rep T) x;'
        if (useUniverses) {
            int l = ParseUtil.getUniverse(ce.thn);
            int r = ParseUtil.getUniverse(ce.els);
            if (l != 0) { //isn't a primitiv type
                int n = leastUpperUniverseBound(l, r);
                ParseUtil.setUniverse(ce, n);
                l = ParseUtil.getElementUniverse(ce.thn);
                r = ParseUtil.getElementUniverse(ce.els);
                n = leastUpperUniverseBound(l, r);
                ParseUtil.setElementUniverse(ce,n);
            }
        }
        //alx-end

        return ce;
    }

    /**
     * Typecheck a new array construction (e.g., <tt>new int[100]</tt>).
     * @param env The current environment.
     * @param na  The expression to check.
     * @return    The checked expression.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkNewArrayExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ NewArrayExpr na
    ) {
        env.resolveType(sig, na.type);
        Type r = na.type;
        checkTypeModifiers(env, r); 
        for (int i = 0; i < na.dims.size(); i++) {
            // (Type)Check the dimension is an integer.
            Expr e = na.dims.elementAt(i);
            Expr newExpr = checkExpr(env, e, Types.intType);
            na.dims.setElementAt(newExpr, i);
            
            // Add a dimension to the array type.
            r = ArrayType.make(r, na.locOpenBrackets[i]);
            checkTypeModifiers(env, r); 
        }
        setType(na, r);

        if (na.init != null) {
            // Check the case where we have something like
            //   new int[] {1, 2, 3}
            na.init = (ArrayInit)checkInit(env, na.init, r);
            //alx: dw set the universe type for array inits and test it
            if (useUniverses) {
                if (ParseUtil.getUniverse(na.init) == 0 && 
                    ParseUtil.getElementUniverse(na.init) != 0) {
                    ParseUtil.setUniverse(na.init, ParseUtil.getUniverse(na));
                }
                checkUniverseAssignability(na,na.init);
            }
            //alx-end
        }
        
        //alx: dw check if universes are used in the static context
        if (useUniverses) {
            checkNoRepInStaticContext(na);
        }
        //alx-end

        return na;
    }

    /**
     * Typecheck a class instantiation (e.g., <tt>new C()</tt>).
     * @param env The current environment.
     * @param ne  The instantiotion to check.
     * @return    The checked instantiation.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected NewInstanceExpr checkNewInstanceExpr(
        /*@ non_null */ Env env,
        /*@ non_null */ NewInstanceExpr ne
    ) {
        /*
         * We handle the "scoping" of ne.type differently depending on
         * whether or not an explicit enclosing instance ptr is
         * provided:
         */
        TypeSig calleeSig;
        if (ne.enclosingInstance==null) {
            // 1.0 case:  new N(...) ...
            env.resolveType(sig,ne.type);
            checkTypeModifiers(env, ne.type);
            calleeSig = TypeSig.getSig(ne.type);
        } else {
            // 1.1 case: E.new I(...) ...

            // Type check E to get class type enclosingType:
            ne.enclosingInstance = checkExpr(env, ne.enclosingInstance);
            TypeSig enclosingInstanceType;
            try {
                enclosingInstanceType =
                    (TypeSig)getType(ne.enclosingInstance);  //@ nowarn Cast;//caught
            } catch (ClassCastException E) {
                ErrorSet.error(ne.enclosingInstance.getStartLoc(),
                               "The enclosing instance supplied in a new"
                               + " expression must be of a class type.");
                enclosingInstanceType = Types.javaLangObject();
            }

            // Check and "resolve" I:
            if (ne.type.name.size() != 1)
                ErrorSet.error(ne.type.getStartLoc(),
                               "Only a simple name can be used after new"
                               + " when an enclosing instance is supplied.");
            Identifier id = ne.type.name.identifierAt(0);
            int idLoc = ne.type.name.locIdAt(0);
            calleeSig = enclosingInstanceType.lookupType(enclosingInstanceType, id, idLoc);
            if (calleeSig==null)
                ErrorSet.fatal(ne.type.getStartLoc(),
                               "No such type: "
                               + enclosingInstanceType.toString()+"$"+id);
            checkTypeModifiers(env, ne.type);
            TypeSig.setSig(ne.type, calleeSig);
        }


        /*
         * Handle remaining type checking/inference for enclosing instance ptr:
         */

        // Get the type of calleeSig's enclosing instance or null if none
        // exists:
        TypeSig enclosingInstanceType = calleeSig.getEnv(false)
            .getEnclosingInstance();

        // If calleeSig has an enclosing instance and the user didn't supply
        // a value for it, try to infer one:
        if (ne.enclosingInstance==null && enclosingInstanceType != null) {
            Expr enclosingInstance =
                env.lookupEnclosingInstance(enclosingInstanceType,
                                            ne.getStartLoc());
            if (enclosingInstance != null) {
                ne.locDot = ne.getStartLoc();
                ne.enclosingInstance = enclosingInstance;
                checkExpr(env, ne.enclosingInstance, enclosingInstanceType);
            }
        }

        if (ne.enclosingInstance != null) {
            if (enclosingInstanceType==null)
                ErrorSet.error(ne.enclosingInstance.getStartLoc(),
                               "An enclosing instance may be provided only "
                               + "when the named instance type is an inner class");
        } else if (enclosingInstanceType != null) {
            ErrorSet.error(ne.getStartLoc(),
                           "No enclosing instance of class "
                           + enclosingInstanceType
                           + " is in scope; an explicit one must be provided"
                           + " when creating instances of inner class "
                           + calleeSig + ".");
        }

        /*
         * The type that will *actually* call the constructor
         * 
         * (matters if the constructor is "protected"!)
         */
        TypeSig caller = sig;

        /*
         * Handle anonymous class case:
         */
        if (ne.anonDecl != null) {
            // Update anonDecl to have proper supertype:
            if (calleeSig.getTypeDecl() instanceof ClassDecl) {
                Assert.notFalse(ne.anonDecl.superClass==null);  //@ nowarn Pre;
                ne.anonDecl.superClass = ne.type;
            } else {
                ne.anonDecl.superInterfaces.addElement(ne.type);
                calleeSig = Types.javaLangObject();
            }

            // Create and check TypeSig for declared type:
            TypeSig T = Types.makeTypeSig(null, env, ne.anonDecl);
            T.typecheck();
            caller = T;
            setType(ne, T);
        } else
            setType(ne, ne.type);


        Type[] argTypes = checkExprVec(env, ne.args);

        if (!(calleeSig.getTypeDecl() instanceof ClassDecl))
            ErrorSet.error(ne.loc, 
                           "Cannot create an instance of an interface");
        else if (Modifiers.isAbstract(calleeSig.getTypeDecl().modifiers)
                 && ne.anonDecl==null)
            ErrorSet.error(ne.loc,
                           "Cannot create an instance of an abstract class");
        else { 
            try {
                ConstructorDecl cd = calleeSig.lookupConstructor(argTypes, caller);
                ne.decl = cd;
            } catch(LookupException e) {
                reportLookupException(e, "constructor", 
                                      Types.printName(calleeSig), ne.loc);
            }
        }
        
        //alx: dw test args uf .ctor
        if (useUniverses) {
            checkNoRepInStaticContext(ne);
            for (int i = 0; i<ne.args.size(); i++) {
                FormalParaDecl decl_arg = ne.decl.args.elementAt(i);
                if (
                    readonlyStdForPureCtor && 
                    ParseUtil.getUniverse(decl_arg)==TagConstants.IMPL_PEER
                ) {
                    ParseUtil.setUniverse(decl_arg, TagConstants.READONLY);
                    checkUniverseAssignability(
                        ne.decl.args.elementAt(i),
                        ne.args.elementAt(i)
                    );
                }
            }
        }
        //alx-end

        return ne;
    }

    /**
     * Typechecks an array reference expression (e.g., <tt>a[3]</tt>). 
     * @param env The current environment.
     * @param r   The expression to check.
     * @return    The checked expression.
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected ArrayRefExpr checkArrayRefExpr(
        /*@ non_null */ Env env, 
        /*@ non_null */ ArrayRefExpr r
    ) {
        r.array = checkExpr(env, r.array);
        Type arrType = getType(r.array);

        if(arrType instanceof ArrayType) {
            setType(r, ((ArrayType)arrType).elemType);
        } else {
            setType(r, Types.errorType);
            if (!Types.isErrorType(arrType)) {
                ErrorSet.error(r.locOpenBracket, "Attempt to index a non-array value");
            }
        }
  
        r.index = checkExpr(env, r.index);
        Type t = getType(r.index);
        Type ndxType = Types.isNumericType(t) ? Types.unaryPromote(t) : t;
        if (
            !Types.isSameType(ndxType, Types.intType) &&
            !Types.isErrorType(ndxType)
        ) { 
            ErrorSet.error(r.locOpenBracket, "Array index is not an integer");
        }

        //alx: dw set universe modifier for ArrayRefExpr, but this is 
        //   never used because for this switch-case the overriding 
        //   method in escjava doesn't call the super implementation
        if (useUniverses) {
            determineUniverseForArrayRefExpr(r);
        }
        //alx-end
        
        return r;
    }

    /**
     * Typechecks <tt>null</tt>.
     * @param x The null expression.
     * @return  The null expression.
     */
    //@ requires x.getTag() == TagConstants.NULLLIT;
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkNullLitExpr(/*@ non_null */ Expr x) {
        //alx: dw use NULLLIT as a universe modifier
        setType(x, Types.nullType);
        if (useUniverses) {
            ParseUtil.setUniverse(x,TagConstants.NULLLIT);
        }
        //alx-end
        return x;
    }

    /**
     * Typechecks long literal expressions (e.g., <tt>29L</tt>).
     * @param x The literal.
     * @return  The literal.
     */
    //@ requires x.getTag() == TagConstants.LONGLIT;
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkLongLitExpr(/*@ non_null */ Expr x) {
        setType(x, Types.longType);
        return x;
    }

    /**
     * Typechecks integers (e.g., <tt>29</tt>).
     * @param x The literal.
     * @return  The literal.
     */
    //@ requires x.getTag() == TagConstants.INTLIT;
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkIntLitExpr(/*@ non_null */ Expr x) {
        setType(x, Types.intType);
        return x;
    }

    /**
     * Typechecks doubles (e.g., <tt>2.9</tt>).
     * @param x The literal.
     * @return  The literal.
     */
    //@ requires x.getTag() == TagConstants.DOUBLELIT;
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkDoubleLitExpr(/*@ non_null */ Expr x) {
        setType(x, Types.doubleType);
        return x;
    }

    /**
     * Typechecks floats (e.g., <tt>2.9F</tt>). 
     * @param x The literal.
     * @return  The literal.
     */
    //@ requires x.getTag() == TagConstants.FLOATLIT;
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkFloatLitExpr(/*@ non_null */ Expr x) {
        setType(x, Types.floatType);
        return x;
    }

    /**
     * Typechecks booleans.
     * @param x The literal.
     * @return  The literal.
     */
    //@ requires x.getTag() == TagConstants.BOOLEANLIT;
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkBooleanLitExpr(Expr x) {
        setType(x, Types.booleanType);
        return x;
    }

    /**
     * Typechecks characters.
     * @param x The literal.
     * @return  The literal.
     */
    //@ requires x.getTag() == TagConstants.CHARLIT;
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkCharLitExpr(/*@ non_null */ Expr x) {
        setType(x, Types.charType);
        return x;
    }

    /**
     * Typechecks strings.
     * @param x The expression to be checked.
     * @return  The checked expression.
     */
    //@ requires x.getTag() == TagConstants.STRINGLIT;
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkStringLitExpr(/*@ non_null */ Expr x) {
        setType(x, Types.javaLangString());
        //alx: dw set to implicit peer
        ParseUtil.setUniverse(x, TagConstants.IMPL_PEER);
        //alx-end
        return x;
    }

    /**
     * Typechecks <tt>this</tt> expression.
     * @param env The current environment.
     * @param e   The expression to check.
     * @return    The checked expression (which was mutated).
     */
    //@ ensures getTypeOrNull(\result) != null;
    protected Expr checkThisExpr(/*@ non_null */ Env env, /*@ non_null */ ThisExpr e) {
             //alx: dw use this like a universe modifier to simplify universeTypeCombiner
             if (useUniverses) {
                 ParseUtil.setUniverse(e, TagConstants.THISEXPR);
        }
             // alx-end

        // It is illegal to use [this] in static methods.
        if (env.isStaticContext() && e.classPrefix == null) {
            ErrorSet.error(
                e.getStartLoc(),
                "Unqualified this cannot be used in static contexts"
            );
        }

        Type referredType = sig;
        if (e.classPrefix != null) {
            env.resolveType(sig, e.classPrefix);
            referredType = e.classPrefix;
            checkTypeModifiers(env, referredType);

            // It is illegal to use [X.this] if we are not within X.
            TypeSig classPrefix = Types.toClassTypeSig(referredType);
            if (classPrefix == null || !env.canAccessInstance(classPrefix)) {
                ErrorSet.error(
                    e.getStartLoc(),
                    "Undefined variable: "
                        + PrettyPrint.inst.toString(referredType) + ".this"
                );
                setType(e, Types.errorType);
                return e;
            }
            
            // alx: dw this of enclosing instances is peer in pure .ctors
            if (useUniverses) {
                if (inCtor && inPure) {
                    ParseUtil.setUniverse(e, TagConstants.READONLY);
                } else {
                    ParseUtil.setUniverse(e, TagConstants.PEER);
                }
            }
            // alx-end
        }

        setType(e, referredType);
        return e;
    }

    // ======================================================================

    //@ requires env != null && od != null;
    //@ requires !(env instanceof EnvForCU);
    //@ requires sig != null;
    protected Type checkObjectDesignator(Env env, ObjectDesignator od) {

        switch(od.getTag()) {
      
            case TagConstants.EXPROBJECTDESIGNATOR: {
                ExprObjectDesignator eod = (ExprObjectDesignator)od;
                eod.expr = checkExpr(env, eod.expr);
                eod.type = getType(eod.expr);
                //alx: dw copy universes of the inner expression
                if (useUniverses && ParseUtil.getUniverse(eod.expr)!=0)
                        ParseUtil.setUniverse(eod,
                                              ParseUtil.getUniverse(eod.expr));
                //alx-end
                return eod.type;
            }

            case TagConstants.TYPEOBJECTDESIGNATOR: {
                TypeObjectDesignator tod = (TypeObjectDesignator)od;
                // Type has been created by disambiguation code, so it is ok.

                Type t = tod.type;
                if(t instanceof TypeName) {
                    t = TypeSig.getSig((TypeName)t);
                    tod.type = t;
                }
                Assert.notFalse(t instanceof TypeSig);
                checkTypeModifiers(env, t);  
                return (TypeSig)t;
            }

            case TagConstants.SUPEROBJECTDESIGNATOR: {
                SuperObjectDesignator sod = (SuperObjectDesignator)od;
                //alx: dw super is peer
                if (useUniverses)
                    ParseUtil.setUniverse(sod,TagConstants.PEER);
                //alx-end

                if(env.isStaticContext()) {
                    ErrorSet.error(sod.locSuper,
                                   "Keyword super cannot be used in a static context");
                    return null;
                }
                else {
                    TypeDecl d = sig.getTypeDecl();
                    if (d instanceof ClassDecl) {
                        TypeName superClass = ((ClassDecl)d).superClass;
                        if (superClass==null) {
                            Assert.notFalse(sig==Types.javaLangObject()); //@ nowarn Pre;
                            ErrorSet.error(sod.locDot, 
                                            "Can't use keyword super in java.lang.Object");
                            return null;
                        }

                        TypeSig ts = TypeSig.getSig(superClass);
                        sod.type = ts;
                        return ts;
                    } else {
                        ErrorSet.error(sod.locDot, 
                                        "Can't use keyword super in an interface");
                        return null;
                    }
                }
            }

            default:
                Assert.fail("Fall thru");
                return null; // dummy return
        }
    }

    // ======================================================================

    /**
     * Return the type of a E1 : L ? R expression given the typechecked Expr's for L
     * and R, as per JLS 15.24.
     *
     * @return null if the given combination is illegal.
     */
    //@ requires leftExpr != null && rightExpr != null;
    private Type tryCondExprMatch(Expr leftExpr, Expr rightExpr) {
        Type leftType = getType(leftExpr);
        Type rightType = getType(rightExpr);
                                          
        /*
         * If the second and third operands have the same type,
         * then than is the type of the conditional expression:
         */
        if (Types.isSameType(leftType, rightType))
            return leftType;


        /*
         * Rules for case where both L and R have numeric types:
         */
        if (Types.isNumericType(leftType)
            && Types.isNumericType(rightType)) {
            /*
             * If one of the operands is of type byte 
             * and the other is of type short, 
             * then the type of the conditional expression is short:
             */
            if (Types.isSameType(leftType, Types.byteType) &&
                Types.isSameType(rightType, Types.shortType))
                return Types.shortType;
            if (Types.isSameType(rightType, Types.byteType) &&
                Types.isSameType(leftType, Types.shortType))
                return Types.shortType;

            /*
             * If one of the operands is of type T where T is byte/short/char, 
             * and the other operand is a constant expression of type int
             * whose value is representable in type T,
             * then the type of the conditional expression is T.
             */
            if ((Types.isSameType(leftType, Types.byteType) ||
                 Types.isSameType(leftType, Types.shortType) ||
                 Types.isSameType(leftType, Types.charType))
                && ConstantExpr.
                constantValueFitsIn(ConstantExpr.eval(rightExpr),
                                    (PrimitiveType)leftType))
                return leftType;
            if ((Types.isSameType(rightType, Types.byteType) ||
                 Types.isSameType(rightType, Types.shortType) ||
                 Types.isSameType(rightType, Types.charType))
                && ConstantExpr.
                constantValueFitsIn(ConstantExpr.eval(leftExpr),
                                    (PrimitiveType)rightType))
                return rightType;


            /*
             * Otherwise, binary numeric promotion (5.6.2) is applied
             * to the operand types, and the type of the conditional
             * expression is the promoted type:
             */
            return Types.binaryNumericPromotion(leftType, rightType);
        }


        /*
         * If one operands is of the null type and the other is 
         * a reference type, then the result type is that reference type
         */
        if (Types.isSameType(leftType, Types.nullType) &&
            Types.isReferenceType(rightType)) 
            return rightType;
        if (Types.isSameType(rightType, Types.nullType) &&
            Types.isReferenceType(leftType)) 
            return leftType;


        /*
         * If the second and third operands are of different reference
         * types, then it must be possible to convert one of the types
         * to the other type (call this latter type T) by assignment
         * conversion (5.2); the type of the conditional expression
         * is T. It is a compile-time error if neither type is
         * assignment compatible with the other type.
         */
        if (Types.isReferenceType(leftType) 
            && Types.isReferenceType(rightType)) {
            if (assignmentConvertable(leftExpr, rightType))
                return rightType;
            if (assignmentConvertable(rightExpr, leftType))
                return leftType;
        }

        // Invalid combination
        return null;

    }

    // ======================================================================

    //@ requires left != null && right != null;
    //@ requires loc != Location.NULL;
    //@ ensures \result != null  ;
    protected Type checkBinaryExpr(int op, Expr left, Expr right, int loc) {
  
        Type leftType = getType(left);
        Type rightType = getType(right);

        switch(op) {
    
            case TagConstants.ADD:
    
                if(Types.isSameType(leftType, Types.javaLangString()) 
                    || Types.isSameType(rightType, Types.javaLangString())) {
      
                    // Operation is string concatenation
                    return Types.javaLangString();
                }
                // else fall thru
      
            case TagConstants.STAR: 
            case TagConstants.DIV: 
            case TagConstants.MOD: 
            case TagConstants.SUB: {
                if(checkNumericType(left)
                    && checkNumericType(right))
                    return Types.binaryNumericPromotion(leftType, rightType);
                else
                    // error recovery
                    return Types.intType;
            } 

            case TagConstants.LSHIFT: 
            case TagConstants.RSHIFT: 
            case TagConstants.URSHIFT:
      
                // Arguments must be primitive integral types
                // result is unary promoted type of left operand
      
                if(checkIntegralType(left) 
                    && checkIntegralType(right))
                    return Types.unaryPromote(leftType);
                else
                    // error recovery
                    return Types.intType;
      
            case TagConstants.LT: 
            case TagConstants.LE: 
            case TagConstants.GT: 
            case TagConstants.GE:
      
                // Arguments must be primitive numeric types
                // result is boolean
      
                checkNumericType(left);
                checkNumericType(right);
                return Types.booleanType;
      
            case TagConstants.BITAND: 
            case TagConstants.BITOR: 
            case TagConstants.BITXOR: {
                if(!Types.isBooleanType(leftType)) {
          
                    // Arguments are primitive integral
                    // binary numeric promotion is performed
                    // type of result is promoted type of operands
          
                    if(checkIntegralType(left)
                        && checkIntegralType(right))
                        return Types.binaryNumericPromotion(leftType, rightType);
                    else
                        // error recovery
                        return Types.intType;
                } 
        
                // else fall thru. Arguments must be boolean, result is boolean
            }
      
            case TagConstants.AND: 
            case TagConstants.OR: {
                // Arguments must be boolean, result is also boolean
                checkType(left, Types.booleanType);
                checkType(right, Types.booleanType);
                return Types.booleanType;
            }
      
            case TagConstants.EQ: 
            case TagConstants.NE:
                if( (Types.isNumericType(leftType) 
                       && Types.isNumericType(rightType))
                      || (!Types.isVoidType(leftType)
                          && Types.isSameType(leftType, rightType))
                      || (Types.isReferenceOrNullType(leftType) 
                          && Types.isReferenceOrNullType(rightType))) {
        
                    if(  Types.isCastable(leftType, rightType) 
                          || Types.isCastable(rightType, leftType)) {
                        // is ok, result boolean
                        return Types.booleanType;
                    } 
                }
      
                // Fall thru to here on error

                if (!Types.isErrorType(leftType) && !Types.isErrorType(rightType)) {
                    ErrorSet.error(loc, 
                               "Invalid types ("+Types.printName(leftType)
                               +" and "+Types.printName(rightType)
                               +") in equality comparison");
                }
                return Types.booleanType;
      
            case TagConstants.ASSIGN:
            case TagConstants.ASGMUL: 
            case TagConstants.ASGDIV: 
            case TagConstants.ASGREM: 
            case TagConstants.ASGADD: 
            case TagConstants.ASGSUB:
            case TagConstants.ASGLSHIFT: 
            case TagConstants.ASGRSHIFT: 
            case TagConstants.ASGURSHIFT:
            case TagConstants.ASGBITAND:
            case TagConstants.ASGBITOR: 
            case TagConstants.ASGBITXOR: {
                if (left instanceof VariableAccess) {
                    //alx: dw check that target is not readonly & 
                    //        field is not rep (if target!=this)
                    if (useUniverses) {
                        FieldAccess fa = (FieldAccess) left;
                        if (ParseUtil.getUniverse(fa.od)==TagConstants.READONLY)
                            ErrorSet.error(fa.getStartLoc(),
                                  "cannot assign to field of readonly target");
                        else if (ParseUtil.getUniverse(fa.decl)==TagConstants.REP && 
                                 ParseUtil.getUniverse(fa.od)!=TagConstants.THISEXPR)
                            ErrorSet.error(fa.getStartLoc(),
                                  "cannot assign to rep field of target "+
                                  "different from this");
                    }
                    GenericVarDecl v = ((VariableAccess)left).decl;
                    if (Modifiers.isFinal(v.modifiers) && 
                        (v instanceof FormalParaDecl ||
                        (v instanceof LocalVarDecl &&
                            ((LocalVarDecl)v).init != null)
                        ))
                        ErrorSet.caution(left.getStartLoc(),
                            "May not assign to a final variable",
                            v.getStartLoc());
                } else if (left instanceof FieldAccess) {
                    GenericVarDecl v = ((FieldAccess)left).decl;
                    // v is null if there was an error such as a field
                    // name that does not exist
                    if (v == Types.lengthFieldDecl) 
                        ErrorSet.error(left.getStartLoc(),
                            "May not assign to array's length field");
                    else if (v != null && Modifiers.isFinal(v.modifiers) &&
                        v instanceof FieldDecl &&
                        ((FieldDecl)v).init != null)
                        ErrorSet.caution(left.getStartLoc(),
                            "May not assign to a final field",
                            v.getStartLoc());

                }
                    // FIXME - need checks of more complicated expressions

                if(!isVariable(left)) {
                    if (!Types.isErrorType(leftType)) ErrorSet.error(loc,
                                   "Left hand side of assignment operator "+
                                   "is not a location");
                } else if(op == TagConstants.ASSIGN) {
                    if(!assignmentConvertable(right, leftType) &&
                        !Types.isErrorType(getType(right)) && !Types.isErrorType(leftType))
                        ErrorSet.error(loc,
                                       "Value of type "+Types.printName(getType(right))
                                       +" cannot be assigned to location of type "
                                       + Types.printName(leftType));
                    //alx: dw check that target is not readonly for 
                    //        array element assignments and test universe
                    //        type assignment
                    if (useUniverses) {
                            if (left instanceof ArrayRefExpr) {
                            ArrayRefExpr ar = (ArrayRefExpr) left;
                            if (ParseUtil.getUniverse(ar.array)==TagConstants.READONLY)
                                ErrorSet.error(ar.getStartLoc(),
                                       "cannot assign to elements of "+
                                       "a readonly array reference");
                            }
                            // test universe types of assignment
                            checkUniverseAssignability(left,right);
                    }
                    //alx-end
                } else {
                    // E1 op= E2 is equivalent to
                    // E1 = (T)(E1 op E2), where T is type of E1
                    int simpleop;
                    switch(op) {
                        case TagConstants.ASGMUL:    simpleop=TagConstants.STAR;    break;
                        case TagConstants.ASGDIV:    simpleop=TagConstants.DIV;     break;
                        case TagConstants.ASGREM:    simpleop=TagConstants.MOD;     break;
                        case TagConstants.ASGADD:    simpleop=TagConstants.ADD;     break;
                        case TagConstants.ASGSUB:    simpleop=TagConstants.SUB;     break;
                        case TagConstants.ASGLSHIFT: simpleop=TagConstants.LSHIFT;  break;
                        case TagConstants.ASGRSHIFT: simpleop=TagConstants.RSHIFT;  break;
                        case TagConstants.ASGURSHIFT:simpleop=TagConstants.URSHIFT; break;
                        case TagConstants.ASGBITAND: simpleop=TagConstants.BITAND;  break;
                        case TagConstants.ASGBITOR:  simpleop=TagConstants.BITOR;   break;
                        case TagConstants.ASGBITXOR: simpleop=TagConstants.BITXOR;  break;
                        default: Assert.fail("Incomplete case"); simpleop=0; // dummy init
                    }
                    Type result = checkBinaryExpr(simpleop, left, right, loc);
                    // Check if result is convertable to leftType
                    if(!Types.isCastable(result, leftType))
                        ErrorSet.error(loc, 
                                       "Result type "+Types.printName(result)
                                       +" of assignment operation cannot be cast to type "
                                       +Types.printName(leftType));
                }  
                // done
                return leftType;
            }

            default:
                Assert.fail("Fall thru, op = "+op); 
                return null;
        }
    }

    // *********************************************************************

    //@ requires e != null;
    static boolean checkIntegralType(Expr e) {
        Type t = getType(e);
        if(Types.isIntegralType(t)) {
            return true;
        } else {
            if (!Types.isErrorType(t))
                ErrorSet.error(e.getStartLoc(), 
                            "Cannot convert "+Types.printName(t)
                            +" to an integral type");
            return false;
        }
    }

    //@ requires e != null;
    static boolean checkNumericType(Expr e) {
        Type t = getType(e);
        if(!Types.isNumericType(t)) {
            if (!Types.isErrorType(t))
                ErrorSet.error(e.getStartLoc(), 
                            "Cannot convert "+Types.printName(t)
                            +" to a numeric type ");
            return false;
        } else {
            return true;
        }    
    }

    //@ requires e != null;
    static boolean isVariable(Expr e) {
        switch(e.getTag()) {
            case TagConstants.ARRAYREFEXPR:
            case TagConstants.VARIABLEACCESS:
            case TagConstants.FIELDACCESS: 
                return true;

            default:
                return false;
        }
    }

    // ======================================================================

    /**
     * Decorates <code>VarInit</code> nodes to point to <code>Type</code> objects.
     */
    //@ private static invariant typeDecoration.decorationType == \type(Type);
    private static /*@non_null*/ ASTDecoration typeDecoration
        = new ASTDecoration("typeDecoration");
    static { 
      initTypeDecoration();
    }
    private /*@helper*/ static void initTypeDecoration()
    {
      //@ set typeDecoration.decorationType = \type(Type);
    }

    //@ requires i != null && t != null;
    public static VarInit setType(VarInit i, Type t) {
        if (t instanceof TypeName)
            t = TypeSig.getSig((TypeName)t);
        typeDecoration.set(i, t);
        return i;
    }

    /**
     * Retrieves the <code>Type</code> of a <code>VarInit</code>.  This type is
     * associated with an expression by the typechecking pass. If the expression does
     * not have an associated type, then null is returned.
     */
    /* FIXME: Chalin - temporarily set to public because it is needed in escjava.AnnotationHandler.makeNonNullExpr(Expr, int) */
    public static Type getTypeOrNull(/*@non_null*/ VarInit i) {
        return (Type)typeDecoration.get(i);
    }

    /**
     * Retrieves the <code>Type</code> of a <code>VarInit</code>.  This type is
     * associated with an expression by the typechecking pass. If the expression does
     * not have an associated type, then <code>Assert.fail</code> is called.
     */
    //@ requires i != null;
    //@ ensures \result != null;
    public static Type getType(VarInit i) {
        Type t = getTypeOrNull(i);
        if (t == null) {
            Assert.fail(
                "getType at " + i.getTag() + " " + PrettyPrint.inst.toString(i) +
                    Location.toString(i.getStartLoc())
            );
        }
        return t;
    }


    // ======================================================================

    /**
     * Decorates <code>BranchStmt</code> nodes to point to labelled <code>Stmt</code>
     * objects.
     */
    //@ private invariant branchDecoration != null;
    //@ private invariant branchDecoration.decorationType == \type(Stmt);
    private static ASTDecoration branchDecoration
        = new ASTDecoration("branchDecoration");

    //@ requires s != null && l != null;
    private static void setBranchLabel(BranchStmt s, Stmt l) {
        Assert.notFalse(branchDecoration.get(s) == null);        //@ nowarn Pre;
        branchDecoration.set(s,l);
    }

    /**
     * Retrieves the <code>Stmt</code> target of a <code>BranchStmt</code>.  This
     * <code>Stmt</code> may be mentioned either explicitly (as in <code>break
     * label;</code>), or implicitly (as in <code>break;</code>) by the
     * <code>BranchStmt</code>.  The correct <code>Stmt</code> target is associated
     * with the <code>BranchStmt</code> by the typechecking pass. This type is
     * associated with an expression by the typechecking pass. If the
     * <code>BranchStmt</code> does not have an associated <code>Stmt</code> target,
     * then <code>Assert.fail</code> is called.
     */
    //@ requires s != null;
    static Stmt getBranchLabel(BranchStmt s) {
        Stmt l = (Stmt)branchDecoration.get(s);
        if(l==null)
            Assert.fail("getBranchLabel failed at "+s.getTag());
        return l;
    }

    // ======================================================================

    //@ requires expr != null && t != null;
    static void checkType(Expr expr, Type t) {
        if(!inst.assignmentConvertable(expr, t)) {
            if (!Types.isErrorType(getType(expr)))
                ErrorSet.error(expr.getStartLoc(), 
                            "Cannot convert "+Types.printName(getType(expr))
                            +" to "+Types.printName(t));
        }
    
    }

    //@ requires e != null && s != null && t != null;
    //@ requires loc != Location.NULL;
    protected static void reportLookupException(LookupException e, 
                                                String s, 
                                                String t, 
                                                int loc) {
        switch(e.reason) {
            case LookupException.NOTFOUND:
                ErrorSet.error(loc, "No such "+s+" in type "+t);
                break;
            case LookupException.AMBIGUOUS:
                ErrorSet.error(loc, "Ambiguous "+s+" for type "+t);
                break;
            case LookupException.BADTYPECOMBO:
                ErrorSet.error(loc, "No "+s+" matching given argument types");
                break;
            case LookupException.NOTACCESSIBLE:
                ErrorSet.error(loc, "Cannot access this "+s);
                break;
            default:
                Assert.fail("Bad lookup exception: "+e.reason);
        }
    }

    /**
     * Checks if Exp e can be assigned to Type t.  This method is here instead of in
     * {@link javafe.tc.Types}, because it needs to mess with constants.
     */ 
    //@ requires e != null && t != null;
    protected boolean assignmentConvertable(Expr e, Type t) {

        Type te = getType(e);

        if(Types.isInvocationConvertable(te, t))
            return true;

        // Check if t is byte/short/char, 
        // and e a constant expression whose value fits in t.

        switch(t.getTag()) {
            case TagConstants.BYTETYPE:
            case TagConstants.SHORTTYPE:
            case TagConstants.CHARTYPE:
                Object val = ConstantExpr.eval(e);
                if(val != null &&
                   ConstantExpr.constantValueFitsIn(val, (PrimitiveType)t))
                    return true;
                else return false;
            default:
                return false;
        }
    }

    // ======================================================================

    //@ requires e != null;
    protected void checkTypeDeclElemPragma(TypeDeclElemPragma e) {
        Assert.fail("Unexpected TypeDeclElemPragma");
    }

    /**
     * Hook to do additional processing on <code>ModifierVec</code>s.  The
     * <code>ASTNode</code> is the parent of the <code>ModifierPragma</code>, and
     * <code>env</code> is the current environment.
     */
    //@ requires env != null;
    protected Env checkModifierPragmaVec(ModifierPragmaVec v, 
                                         ASTNode ctxt, 
                                         Env env) {
        if (v != null)
          for(int i = 0; i < v.size(); i++) {
            env = checkModifierPragma(v.elementAt(i), ctxt, env);
              }
          return env;
    }

    /**
     * Hook to do additional processing on <code>Modifier</code>s.  The
     * <code>ASTNode</code> is the parent of the <code>ModifierPragma</code>, and
     * <code>env</code> is the current environment.
     * @return true if pragma should be deleted
     */
    //@ requires p != null && env != null;
    protected Env checkModifierPragma(ModifierPragma p, ASTNode ctxt, Env env) {
        // Assert.fail("Unexpected ModifierPragma");
        return env;
    }

    //@ requires e != null && s != null;
    protected Env checkStmtPragma(Env e, StmtPragma s) {
        Assert.fail("Unexpected StmtPragma");
        return e;
    }

    //@ requires env != null;
    protected Env checkTypeModifierPragmaVec(TypeModifierPragmaVec v, 
                                              ASTNode ctxt, 
                                              Env env) {
        if(v != null)
            for(int i=0; i<v.size(); i++)
                env = checkTypeModifierPragma(v.elementAt(i), ctxt, env);
        return env;
    }
    
    //@ requires p != null && env != null;
    protected Env checkTypeModifierPragma(TypeModifierPragma p,
                                           ASTNode ctxt,
                                           Env env) {
        Assert.fail("Unexpected TypeModifierPragma");
        return env;
    }

    /**
     * This may be called more than once on a Type t.
     */
    //@ requires t != null;
    protected Env checkTypeModifiers(Env env, Type t) {
        // don't know context for type, so pull it out of the type's decorations.
        if (env == null) {
            env = (Env)Env.typeEnv.get(t);
        }        
        Assert.notFalse(env != null);  //@ nowarn Pre;
        checkTypeModifierPragmaVec(t.tmodifiers, t, env);
        if (t instanceof ArrayType) {
            env = checkTypeModifiers(env, ((ArrayType)t).elemType);
        }
        return env;
    }

    //alx: dw needs overriding method
    //no assumptions on the parameter, TODO: what does that method
    protected boolean isPure(RoutineDecl rd) {
            return false;
    }
    //alx-end
    
    //alx: dw TODO what does that method
    protected static int universeTypeCombiner(int l, int r) {
            switch (l) {
            case TagConstants.THISEXPR:        //dw: special case for l=peer=this
            {
                    return r;
            }
            case TagConstants.PEER:
            case TagConstants.IMPL_PEER:
            {
                    if (r==TagConstants.PEER || r==TagConstants.IMPL_PEER)
                    return TagConstants.PEER;
                    return TagConstants.READONLY;
            }
            case TagConstants.REP:
            {
                    if (r==TagConstants.PEER || r==TagConstants.IMPL_PEER)
                    return TagConstants.REP;
                    return TagConstants.READONLY;
            }
            case TagConstants.READONLY:
            {
                    return TagConstants.READONLY;
            }
            }
            
            return 0;
    }
    //alx-end
    
    //alx: dw TODO: what does the method
    protected static int universeTypeCombiner(/*@ non_null @*/ ASTNode left, 
                                              /*@ non_null @*/ ASTNode right) {
            return universeTypeCombiner(ParseUtil.getUniverse(left),
                                    ParseUtil.getUniverse(right)); 
    }
    //alx-end
    
    //alx: dw TODO: what does the method
    protected void checkNoRepInStaticContext(/*@ non_null @*/ ASTNode n) {
            if (inStatic && ParseUtil.getUniverse(n)==TagConstants.REP)
            ErrorSet.error(n.getStartLoc(),
                           "usage of rep not allowed in static contexts");
    }
    //alx-end
    
    
    //alx: dw TODO: what does the method
    protected void checkUniverseForField(/*@ non_null @*/ GenericVarDecl gvd) {
            int mod=ParseUtil.getUniverse(gvd);
            if (mod!=0 && Modifiers.isStatic(gvd.modifiers) && 
            mod!=TagConstants.READONLY) {
            if (mod==TagConstants.REP)
                ErrorSet.error(gvd.getStartLoc(),
                               "static fields cannot be of universe type rep");
            if (!impl_peerInStaticCautionThrown && 
                mod==TagConstants.IMPL_PEER) {
                ErrorSet.caution(gvd.getStartLoc(),
                                 "using 'implicit peer' as default for "+
                                 "static fields, but should be readonly");
                impl_peerInStaticCautionThrown=true;
            }
            }
    }
    //alx-end
    
    //alx: dw TODO: where is it used?
    boolean impl_peerInStaticCautionThrown = false;
    //alx-end
    
    //alx: dw TODO whad does the method?
    public static void copyUniverses(/*@ non_null @*/ ASTNode dest,
                                     /*@ non_null @*/ ASTNode source) {
            int r = ParseUtil.getUniverse(source);
            if (r!=0) {
            ParseUtil.setUniverse(dest,r);
            int e = ParseUtil.getElementUniverse(source);
            if (e!=0)
                ParseUtil.setElementUniverse(dest,e);
            }
    }
    //alx-end
    
    
    //alx: dw the first common supertype (used for array inits and ?: )
    //TODO: what are the specs here?
    public static int leastUpperUniverseBound(int l, int r) {
            if (l==TagConstants.THISEXPR || l==TagConstants.IMPL_PEER)
            l=TagConstants.PEER;
            if (r==TagConstants.THISEXPR || r==TagConstants.IMPL_PEER)
            r=TagConstants.PEER;
            
            if (l==r)
            return r;
            else if (l==TagConstants.NULLLIT)
            return r;
            else if (r==TagConstants.NULLLIT)
            return l;
            else
            return TagConstants.READONLY;
    }
    //alx-end
    
    
    //alx: dw determine & set universe type of field access by applying 
    //        universeTypeCombiner. fa.od must not be null!!!
    //TODO: is the spec OK?
    public static void determineUniverseForFieldAccess(
                                    /*@ non_null @*/ FieldAccess fa) {
            //if target has no universe modifier, set it to impl_peer
            if (fa.od!=null && ParseUtil.getUniverse(fa.od)==0)
            ParseUtil.setUniverse(fa.od,TagConstants.IMPL_PEER);
            //if decl has no universe modifier, set to default
            if (!(fa.decl.type instanceof PrimitiveType) && 
            ParseUtil.getUniverse(fa.decl)==0)
            ParseUtil.setUniverse(fa.decl,
                                  new int[] {0,0},
                                  fa.decl.type,
                                  Location.NULL);
            //use universeTypeCombiner
            if (ParseUtil.getUniverse(fa.decl)!=0) {
            ParseUtil.setUniverse(fa,universeTypeCombiner(fa.od,fa.decl));
            if (ParseUtil.getElementUniverse(fa.decl)!=0)
                ParseUtil.setElementUniverse(fa, 
                                    ParseUtil.getElementUniverse(fa.decl));
            }
    }
    //alx-end
    
    //alx: dw TODO what does the method?
    public void determineUniverseForMethodInvocation(
                                    /*@ non_null @*/ MethodInvocation mi) {
            //if target has no universe modifier, set it to impl_peer
            if (mi.od!=null && ParseUtil.getUniverse(mi.od)==0)
            ParseUtil.setUniverse(mi.od,TagConstants.IMPL_PEER);
            int od = ParseUtil.getUniverse(mi.od);
            //if decl has no universe modifier, set to default
            if (!(mi.decl.returnType instanceof PrimitiveType) && 
            ParseUtil.getUniverse(mi.decl)==0)
              ParseUtil.setUniverse(mi.decl,
                                    new int[] {0,0},
                                    mi.decl.returnType,
                                    Location.NULL);
            //use universeTypeCombiner
            if (ParseUtil.getUniverse(mi.decl)!=0) { 
            ParseUtil.setUniverse(mi,universeTypeCombiner(mi.od,mi.decl));
            if (ParseUtil.getElementUniverse(mi.decl)!=0)
                ParseUtil.setElementUniverse(mi,
                                    ParseUtil.getElementUniverse(mi.decl));
                   
            }
            boolean isPureMeth=isPure(mi.decl);
            if (od==TagConstants.READONLY && !isPureMeth)
            ErrorSet.error(mi.getStartLoc(),
                   "only pure methods can be called on readonly targets");
            boolean repIsError = od!=TagConstants.THISEXPR;
            for (int i = 0; i<mi.args.size(); i++) {
            int decl_universe_arg;
            FormalParaDecl decl_arg=mi.decl.args.elementAt(i);
            if (decl_arg.type instanceof PrimitiveType)
                continue;
            Expr actual_arg=mi.args.elementAt(i);
            if (ParseUtil.getUniverse(decl_arg)==0 && !isPureMeth)
                ParseUtil.setUniverse(decl_arg,
                                      new int[] {0,0},
                                      decl_arg.type,
                                      Location.NULL);
            else if (isPureMeth) {
                decl_universe_arg = ParseUtil.getUniverse(decl_arg);
                if (decl_universe_arg!=TagConstants.READONLY) {
                    if (decl_universe_arg!=TagConstants.IMPL_PEER && 
                        decl_universe_arg!=0)
                        ErrorSet.error(decl_arg.getStartLoc(),
                             "parameters of pure methods have to be readonly");
                    ParseUtil.setUniverse(decl_arg,TagConstants.READONLY);
                }
            }
            decl_universe_arg = ParseUtil.getUniverse(decl_arg);
            //if the target is rep and a method is defined as foo(peer T x), 
            //then the actual argument has to be 
            // rep T. (universeTypeCombiner: rep*peer=rep)
            if (od==TagConstants.REP && 
                (decl_universe_arg==TagConstants.PEER || 
                 decl_universe_arg==TagConstants.IMPL_PEER)) {
                if (ParseUtil.getUniverse(actual_arg)!=TagConstants.REP)
                    ErrorSet.error(actual_arg.getStartLoc(),
                                   "the argument has to be rep: rep*peer=rep");
            }
            else {
                checkUniverseAssignability(decl_arg,actual_arg);
                if (repIsError && decl_universe_arg==TagConstants.REP) {
                    ErrorSet.error(mi.getStartLoc(),
                                   "methods with rep arguments can only be "+
                                   "called on target this");
                    repIsError=false;
                }
            }
            }
    }
    //alx-end
    
    //alx: dw TODO what does the method
    public static void determineUniverseForArrayRefExpr(
                              /*@ non_null @*/ ArrayRefExpr r) {
            ParseUtil.setUniverse(r,universeTypeCombiner(
                       ParseUtil.getUniverse(r.array),
                       ParseUtil.getElementUniverse(r.array)));
    }
    //alx-end
}

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */
