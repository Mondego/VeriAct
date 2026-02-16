package escjava.sortedProver.cvc3;

//import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import javafe.util.Assert;
import javafe.util.ErrorSet;
import javafe.util.Info;

import escjava.ast.TagConstants;
import escjava.sortedProver.EscNodeBuilder;
import escjava.sortedProver.Lifter;

import cvc3.*;


/*@ non_null_by_default @*/
public class Cvc3NodeBuilder extends EscNodeBuilder {

    //
    /// debug
    //
    final static private boolean printQuery = false;


    public final boolean optMultiTriggers;
    public final boolean optManualTriggers;
    public final boolean optNonnullelements;
    public final boolean optIsallocated;
    public final boolean optBuiltinTrans;
    public final boolean optUseClassLiteral;
    public final boolean optUseDatatype;
    public final boolean optTest;

    //
    /// debug
    //
    // can't use cvc3's preprocessing, as it might remove expressions
    // which we need for label relevance analysis.
    //
    // tried to use some rewriting here, but:
    // - wasn't effective in terms of performance gain
    // - label propagation during simplification is tricky,
    //   and the current code doesn't get it quite right.
    //   that is, some relevant labels might be lost,
    //   and thus counterexamples may be missed.
    // rewrite using simple rules
    final static private boolean rewrite = false;
    // rewrite using cvc3's state
    final static private boolean rewriteCvc3 = false;


    // the cvc instance
    final private cvc3.ValidityChecker prover;

    final private Cvc3Prover cvcProver;


    //
    // sorts
    //

    // for internal errors
    final public Type typeError;

    // the cvc types mirroring the sorts used by EscJava / NodeBuilder
    final public Type typePred;
    final public Type typeValue;
    final public Type typeBool;
    final public Type typeInt;
    final public Type typeReal;
    final public Type typeRef;
    final public Type typeString;
    final public Op isClassType;
    final public Op isArrayType;

    final public Type typeShape;
    final public Type typeTime;
    final public Type typeType;
    final public Type typeLock;

    final public Type typeField;
    final public Type typeBoolField;
    final public Type typeIntField;
    final public Type typeRealField;
    final public Type typeRefField;
    final public Type typeArray;
    final public Type typeElems;
    final public Type typeLockSet;


    // predefined cvc operators

    // casting between cvc types
    final public Op refToType;
    final public Op typeToRef;

    final public Op valueToBool;
    final public Op valueToInt;
    final public Op valueToReal;
    final public Op valueToRef;
    final public Op boolToValue;
    final public Op intToValue;
    final public Op realToValue;
    final public Op refToValue;
    final public Op isBoolValue;
    final public Op isIntValue;
    final public Op isRealValue;
    final public Op isRefValue;
    /*
    final public Op fieldToBoolField;
    final public Op fieldToIntField;
    final public Op fieldToRealField;
    final public Op fieldToRefField;
    final public Op boolFieldToField;
    final public Op intFieldToField;
    final public Op realFieldToField;
    final public Op refFieldToField;
    */
    final public Op[] casts;

    // field mapping functions specialized for the field instances
    final public Op asFieldBool;
    final public Op asFieldInt;
    final public Op asFieldReal;
    final public Op asFieldRef;
    /*
    final public Op fClosedTimeBool;
    final public Op fClosedTimeInt;
    final public Op fClosedTimeReal;
    final public Op fClosedTimeRef;
    */

    // predefined EscJava expressions
    final public Cvc3Ref escNull;
    final public Cvc3Pred escTrue;
    final public Cvc3Pred escFalse;
    final public Cvc3Bool escBTrue;
    final public Cvc3Bool escBFalse;


    // EscJava symbols, that for some reason are not defined in EscNodeBuilder
    // types
    public FnSymbol getTypeSymbol(int tag) {
	return registerConstant(escjava.ast.TagConstants.toString(tag), sortType);
    }

    final public FnSymbol symTArray = getTypeSymbol(TagConstants.ARRAYTYPE);
    final public FnSymbol symTBoolean = getTypeSymbol(TagConstants.BOOLEANTYPE);
    final public FnSymbol symTChar = getTypeSymbol(TagConstants.CHARTYPE);
    final public FnSymbol symTByte = getTypeSymbol(TagConstants.BYTETYPE);
    final public FnSymbol symTShort = getTypeSymbol(TagConstants.SHORTTYPE);
    final public FnSymbol symTInt = getTypeSymbol(TagConstants.INTTYPE);
    final public FnSymbol symTLong = getTypeSymbol(TagConstants.LONGTYPE);
    final public FnSymbol symTFloat = getTypeSymbol(TagConstants.FLOATTYPE);
    final public FnSymbol symTDouble = getTypeSymbol(TagConstants.DOUBLETYPE);
    final public FnSymbol symTBigInt = getTypeSymbol(TagConstants.BIGINTTYPE);
    //:TODO: final public FnSymbol symTReal = getTypeSymbol(TagConstants.REALTYPE);
    final public FnSymbol symTVoid = getTypeSymbol(TagConstants.VOIDTYPE);
    //???FnSymbol symTType = getTypeSymbol(TagConstants.TYPE);
    // special values
    //:TODO: not as hard-coded string - but how?
    final public FnSymbol symTObject = registerConstant("T_java.lang.Object", sortType);
    final public FnSymbol symTClass = registerConstant("T_java.lang.Class", sortType);
    final public FnSymbol symTException = registerConstant("T_java.lang.Exception", sortType);
    final public FnSymbol symTThrowable = registerConstant("T_java.lang.Throwable", sortType);
    final public FnSymbol symTCloneable = registerConstant("T_java.lang.Cloneable", sortType);
    final public FnSymbol symTComparable = registerConstant("T_java.lang.Comparable", sortType);
    final public FnSymbol symTSerializable = registerConstant("T_java.io.Serializable", sortType);
    final public FnSymbol symTString = registerConstant("T_java.lang.String", sortType);
    final public FnSymbol symTCharSequence = registerConstant("T_java.lang.CharSequence", sortType);

    final public FnSymbol symAlloc = registerConstant("alloc", sortTime);
    final public FnSymbol symvAllocTime = registerFnSymbol(escjava.ast.TagConstants.toString(TagConstants.VALLOCTIME),
							   new Sort[] { sortRef }, sortTime, TagConstants.VALLOCTIME);

    final public FnSymbol symLockPrec = registerFnSymbol("lprec", new Sort[] { sortRef }, sortReal, 0);
    final public FnSymbol symIntegralDiv = registerFnSymbol(escjava.ast.TagConstants.toString(TagConstants.INTEGRALDIV),
							    new Sort[] { sortInt, sortInt }, sortInt,
							    TagConstants.INTEGRALDIV);
    final public FnSymbol symIntegralMod = registerFnSymbol(escjava.ast.TagConstants.toString(TagConstants.INTEGRALMOD),
							    new Sort[] { sortInt, sortInt }, sortInt,
							    TagConstants.INTEGRALMOD);
    final public FnSymbol symFloatingDiv = registerFnSymbol(escjava.ast.TagConstants.toString(TagConstants.FLOATINGDIV),
							    new Sort[] { sortInt, sortInt }, sortInt,
							    TagConstants.FLOATINGDIV);
    final public FnSymbol symFloatingMod = registerFnSymbol(escjava.ast.TagConstants.toString(TagConstants.FLOATINGMOD),
							    new Sort[] { sortInt, sortInt }, sortInt,
							    TagConstants.FLOATINGMOD);
    final public FnSymbol symASR32 = registerFnSymbol(escjava.ast.TagConstants.toString(TagConstants.INTSHIFTR),
							    new Sort[] { sortInt, sortInt }, sortInt,
							    TagConstants.INTSHIFTR);
    final public FnSymbol symSL32 = registerFnSymbol(escjava.ast.TagConstants.toString(TagConstants.INTSHIFTL),
							    new Sort[] { sortInt, sortInt }, sortInt,
							    TagConstants.INTSHIFTL);
    final public FnSymbol symUSR32 = registerFnSymbol(escjava.ast.TagConstants.toString(TagConstants.INTSHIFTRU),
							    new Sort[] { sortInt, sortInt }, sortInt,
							    TagConstants.INTSHIFTRU);
    final public FnSymbol symASR64 = registerFnSymbol(escjava.ast.TagConstants.toString(TagConstants.LONGSHIFTR),
							    new Sort[] { sortInt, sortInt }, sortInt,
							    TagConstants.LONGSHIFTR);
    final public FnSymbol symSL64 = registerFnSymbol(escjava.ast.TagConstants.toString(TagConstants.LONGSHIFTL),
							    new Sort[] { sortInt, sortInt }, sortInt,
							    TagConstants.LONGSHIFTL);
    final public FnSymbol symUSR64 = registerFnSymbol(escjava.ast.TagConstants.toString(TagConstants.LONGSHIFTRU),
							    new Sort[] { sortInt, sortInt }, sortInt,
							    TagConstants.LONGSHIFTRU);

    // only created for simplicity, to be used in Cvc3BackgroundPredicate
    // instead of the Ops of the same name create above
    final public FnSymbol symClassType = registerFnSymbol("__cvc3_classType", new Sort[] { sortInt }, sortType, 0);
    final public FnSymbol symRefValue = registerFnSymbol("__cvc3_refValue", new Sort[] { sortRef }, sortValue, 0);
    final public PredSymbol symIsClassType = registerPredSymbol("__cvc3_isClassType", new Sort[] { sortType }, 0);
    final public PredSymbol symIsArrayType = registerPredSymbol("__cvc3_isArrayType", new Sort[] { sortType }, 0);
    final public PredSymbol symIsBoolValue = registerPredSymbol("__cvc3_isBoolValue", new Sort[] { sortBool }, 0);
    final public PredSymbol symIsIntValue = registerPredSymbol("__cvc3_isIntValue", new Sort[] { sortInt }, 0);
    final public PredSymbol symIsRealValue = registerPredSymbol("__cvc3_isRealValue", new Sort[] { sortReal }, 0);
    final public PredSymbol symIsRefValue = registerPredSymbol("__cvc3_isRefValue", new Sort[] { sortRef }, 0);

    //
    // registries / mappings (for memoization)
    //

    // EscJava sort -> cvc type
    final private HashMap typeMap = new HashMap();

    // EscJava symbol -> cvc constant/operator
    final private HashMap symbolMap = new HashMap();

    // EscJava QuantVar -> cvc bound var
    final private HashMap varsMap = new HashMap();

    // cvc expr <-> EscJava node (STerm, SValue, SRef, ...)
    // 
    final private HashMap escMap = new HashMap();
    final private HashMap exprMap = new HashMap();

    // predefined mapping of symbol names to cvc3 names
    final private HashMap predefinedNames = new HashMap();

    // counter for java types introduced for java classes
    private int classId = 0;


    //
    // methods
    //


    public Cvc3NodeBuilder(cvc3.ValidityChecker prover, Cvc3Prover cvcProver,
                           boolean multiTriggers, boolean manualTriggers,
                           boolean nonnullelements, boolean isallocated, boolean builtinTrans,
                           boolean useClassLiteral, boolean useDatatype, boolean test)
	throws cvc3.Cvc3Exception {

	this.prover = prover;
	this.cvcProver = cvcProver;
	this.optMultiTriggers = multiTriggers;
        this.optNonnullelements = nonnullelements;
        this.optIsallocated = isallocated;
        this.optBuiltinTrans = builtinTrans;
	this.optManualTriggers = manualTriggers;
        this.optUseClassLiteral = useClassLiteral;
        this.optUseDatatype = useDatatype;
        this.optTest = test;

	
	// sorts
	typeError = prover.createType("error");
	typePred = prover.boolType();
	typeBool = prover.bitvecType(1);
	typeInt = prover.intType();
	typeReal = prover.realType();
	typeRef = prover.createType("Ref");
	//typeRef = prover.createType("Ref", typeInt);
	typeString = typeRef;
	typeShape = prover.createType("Shape");
        // optional: model time as real,
        // but, surprisingly, less efficient in practice
	//typeTime = typeReal;
	typeTime = typeInt;
	typeLock = typeRef;


	// expressions
	escNull = new Cvc3Ref(prover.varExpr("null", typeRef));
	//escNull = new Cvc3Ref(prover.varExpr("null", typeRef, prover.ratExpr(0, 1)));
	escTrue = new Cvc3Pred(prover.trueExpr());
	escFalse = new Cvc3Pred(prover.falseExpr());
	escBTrue = new Cvc3Bool(prover.newBVConstExpr("1"));
	escBFalse = new Cvc3Bool(prover.newBVConstExpr("0"));

        if (optUseDatatype) {
            // set up JavaType as an inductive datatype
            String name = "JavaType";

            String[] constructors = new String[]
                { "boolType", "charType", "byteType", "shortType", "intType",
                  "longType", "floatType", "doubleType", "voidType",
                  "classType", "arrayType" };

            String[][] selectors = new String[11][];
            selectors[0] = new String[0];
            selectors[1] = new String[0];
            selectors[2] = new String[0];
            selectors[3] = new String[0];
            selectors[4] = new String[0];
            selectors[5] = new String[0];
            selectors[6] = new String[0];
            selectors[7] = new String[0];
            selectors[8] = new String[0];
            selectors[9] = new String[] { "classId" };
            selectors[10] = new String[] { "elemType" };

            Expr[][] types = new Expr[11][];
            types[0] = new Expr[0];
            types[1] = new Expr[0];
            types[2] = new Expr[0];
            types[3] = new Expr[0];
            types[4] = new Expr[0];
            types[5] = new Expr[0];
            types[6] = new Expr[0];
            types[7] = new Expr[0];
            types[8] = new Expr[0];
            types[9] = new Expr[] { typeInt.getExpr() };
            types[10] = new Expr[] { prover.stringExpr(name) };

            typeType = prover.dataType(name, constructors, selectors, types);

            // hack to get arrayType constructor/selector
            // :TODO: unify with setupSymbols segment
            List args = new ArrayList();
            args.add(prover.ratExpr(0));
            Expr expr = prover.datatypeConsExpr("classType", args);
            isClassType = prover.datatypeTestExpr("classType", expr).getOp();

            args = new ArrayList();
            args.add(prover.datatypeConsExpr("boolType", new ArrayList()));
            expr = prover.datatypeConsExpr("arrayType", args);
            isArrayType = prover.datatypeTestExpr("arrayType", expr).getOp();
        } else {
            typeType = prover.createType("JavaType");

            isClassType = null;
            isArrayType = null;
        }

        if (optUseDatatype) {
            // set up JavaValue as an inductive datatype
            String name = "JavaValue";

            String[] constructors = new String[]
                { "javaBool", "javaInt", "javaReal", "javaRef" };

            String[][] selectors = new String[4][];
            selectors[0] = new String[] { "valueToBool" };
            selectors[1] = new String[] { "valueToInt" };
            selectors[2] = new String[] { "valueToReal" };
            selectors[3] = new String[] { "valueToRef" };

            Expr[][] types = new Expr[4][];
            types[0] = new Expr[] { typeBool.getExpr() };
            types[1] = new Expr[] { typeInt.getExpr() };
            types[2] = new Expr[] { typeReal.getExpr() };
            types[3] = new Expr[] { typeRef.getExpr() };

            typeValue = prover.dataType(name, constructors, selectors, types);

            // hacks to get constructors, selectors, testers ...
            List args = new ArrayList();
            args.add(prover.newBVConstExpr("0"));
            Expr expr = prover.datatypeConsExpr("javaBool", args);
            boolToValue = expr.getOp();
            valueToBool = prover.datatypeSelExpr("valueToBool", expr).getOp();
            isBoolValue = prover.datatypeTestExpr("javaBool", expr).getOp();

            args = new ArrayList();
            args.add(prover.ratExpr(0));
            expr = prover.datatypeConsExpr("javaInt", args);
            intToValue = expr.getOp();
            valueToInt = prover.datatypeSelExpr("valueToInt", expr).getOp();
            isIntValue = prover.datatypeTestExpr("javaInt", expr).getOp();

            args = new ArrayList();
            args.add(prover.ratExpr(0));
            expr = prover.datatypeConsExpr("javaReal", args);
            realToValue = expr.getOp();
            valueToReal = prover.datatypeSelExpr("valueToReal", expr).getOp();
            isRealValue = prover.datatypeTestExpr("javaReal", expr).getOp();

            args = new ArrayList();
            args.add(escNull.getExpr());
            expr = prover.datatypeConsExpr("javaRef", args);
            refToValue = expr.getOp();
            valueToRef = prover.datatypeSelExpr("valueToRef", expr).getOp();
            isRefValue = prover.datatypeTestExpr("javaRef", expr).getOp();
        } else {
            typeValue = prover.createType("JavaValue");

            boolToValue = prover.createOp("javaBool", prover.funType(typeBool, typeValue));  
            intToValue = prover.createOp("javaInt", prover.funType(typeInt, typeValue));  
            realToValue = prover.createOp("javaReal", prover.funType(typeReal, typeValue));  
            refToValue = prover.createOp("javaRef", prover.funType(typeRef, typeValue));
            valueToBool = prover.createOp("valueToBool", prover.funType(typeValue, typeBool));  
            valueToInt = prover.createOp("valueToInt", prover.funType(typeValue, typeInt));
            valueToReal = prover.createOp("valueToReal", prover.funType(typeValue, typeReal));
            valueToRef = prover.createOp("valueToRef", prover.funType(typeValue, typeRef));
            isBoolValue = null;
            isIntValue = null;
            isRealValue = null;
            isRefValue = null;
        }

        refToType = prover.createOp("refToClassType", prover.funType(typeRef, typeType));
        typeToRef = prover.createOp("classTypeToRef", prover.funType(typeType, typeRef));

	typeField = prover.createType("Field");
	//typeField = prover.arrayType(typeValue, typeValue);
	typeBoolField = prover.arrayType(typeRef, typeBool);
	typeIntField = prover.arrayType(typeRef, typeInt);
	typeRealField = prover.arrayType(typeRef, typeReal);
	typeRefField = prover.arrayType(typeRef, typeRef);

	typeArray = prover.arrayType(typeInt, typeValue);
	typeElems = prover.arrayType(typeRef, typeArray);
	typeLockSet = prover.arrayType(typeLock, typeBool);
	

	/*
	fieldToBoolField = prover.createOp("fieldToBoolField", prover.funType(typeField, typeBoolField));
	fieldToIntField = prover.createOp("fieldToIntField", prover.funType(typeField, typeIntField));
	fieldToRealField = prover.createOp("fieldToRealField", prover.funType(typeField, typeRealField));
	fieldToRefField = prover.createOp("fieldToRefField", prover.funType(typeField, typeRefField));
	boolFieldToField = prover.createOp("boolFieldToField", prover.funType(typeBoolField, typeField));
	intFieldToField = prover.createOp("intFieldToField", prover.funType(typeIntField, typeField));
	realFieldToField = prover.createOp("realFieldToField", prover.funType(typeRealField, typeField));
	refFieldToField = prover.createOp("refFieldToField", prover.funType(typeRefField, typeField));
	*/
	asFieldBool = prover.createOp("asFieldBool",
	  prover.funType(Arrays.asList (new Type[] { typeBoolField, typeType }), typeBoolField));
	asFieldInt = prover.createOp("asFieldInt",
	  prover.funType(Arrays.asList (new Type[] { typeIntField, typeType }), typeIntField));
	asFieldReal = prover.createOp("asFieldReal",
	  prover.funType(Arrays.asList (new Type[] { typeRealField, typeType }), typeRealField));
	asFieldRef = prover.createOp("asFieldRef",
	  prover.funType(Arrays.asList (new Type[] { typeRefField, typeType }), typeRefField));
	/*	fClosedTimeBool = prover.createOp("fClosedTimeBool", prover.funType(typeBoolField, typeTime));
	fClosedTimeInt = prover.createOp("fClosedTimeInt", prover.funType(typeIntField, typeTime));
	fClosedTimeReal = prover.createOp("fClosedTimeReal", prover.funType(typeRealField, typeTime));
	fClosedTimeRef = prover.createOp("fClosedTimeRef", prover.funType(typeRefField, typeTime));
	*/
	casts = new Op[] {
	    refToType, typeToRef,
	    valueToBool, valueToInt, valueToReal, valueToRef,
	    boolToValue, intToValue, realToValue, refToValue//,
	    //fieldToBoolField, fieldToIntField, fieldToRealField, fieldToRefField,
	    //boolFieldToField, intFieldToField, realFieldToField, refFieldToField
	};
    }

    public void setup() throws cvc3.Cvc3Exception {
        setupPredefinedNames();
	setupTypes();
        // setupSymbols must come after setupPredefinedNames
        // due to redefinition of symFClosedTime
	setupSymbols();
    }


    // mapping: escjava sorts -> cvc types
    protected void setupTypes() {
	typeMap.put(sortPred.name, typePred);
	typeMap.put(sortValue.name, typeValue);
	typeMap.put(sortBool.name, typeBool);
	typeMap.put(sortInt.name, typeInt);
	typeMap.put(sortReal.name, typeReal);
	typeMap.put(sortRef.name, typeRef);
	typeMap.put(sortString.name, typeString);

	typeMap.put(sortShape.name, typeShape);
	typeMap.put(sortTime.name, typeTime);
	typeMap.put(sortType.name, typeType);
	typeMap.put(sortLock.name, typeLock);

	typeMap.put(sortField.name, typeField);
	typeMap.put(sortIntField.name, typeIntField);
	typeMap.put(sortRealField.name, typeRealField);
	typeMap.put(sortBoolField.name, typeBoolField);
	typeMap.put(sortOwner.name, typeRefField);

	typeMap.put(sortArrayValue.name, typeArray);
	typeMap.put(sortElems.name, typeElems);
	typeMap.put(sortLockSet.name, typeLockSet);
    }

    // x2y(y2x(y)) = y
    //
    // adding y2x(x2y(x)) = x as well is unsound,
    // as then we don't have embeddings of int/real/bool/ref into value,
    // but bijections, which also make int and bool isomorphic
    protected void setupCast(Op x2y, Op y2x, Expr x, Expr y) throws cvc3.Cvc3Exception {
	if (printQuery) System.out.println("%% Cvc3NodeBuilder.setupCast:");

	List vars = Arrays.asList(new Expr[] { y } );
	Expr eq = prover.eqExpr(prover.funExpr(x2y, prover.funExpr(y2x, y)), y);
	Expr p = prover.forallExpr(vars, eq, prover.funExpr(y2x, y));

	if (printQuery) {
	    System.out.println("ASSERT (" + p.toString() + ");");
	}
	cvcProver.declareAxiom(new Cvc3Pred(p));
    }

    // setup casting embeddings between types,
    // i.e. between value, the super type,
    // and its subtypes bool/int/real/ref,
    // and also between type and ref for some reflection vcs
    protected void setupCast() throws cvc3.Cvc3Exception {
	/*
	Expr vt = prover.varExpr("t", typeType);
	Expr vv = prover.varExpr("v", typeValue);
	Expr vb = prover.varExpr("b", typeBool);
	Expr vi = prover.varExpr("i", typeInt);
	Expr vr = prover.varExpr("r", typeReal);
	Expr vref = prover.varExpr("ref", typeRef);
	Expr vf = prover.varExpr("f", typeField);
	Expr vfb = prover.varExpr("fb", typeBoolField);
	Expr vfi = prover.varExpr("fi", typeIntField);
	Expr vfr = prover.varExpr("fr", typeRealField);
	Expr vfref = prover.varExpr("fref", typeRefField);
	*/
        Expr vt = prover.boundVarExpr("t", "t_" + typeType.toString(), typeType);
        Expr vv = prover.boundVarExpr("v", "v_" + typeValue.toString(), typeValue);
        Expr vb = prover.boundVarExpr("b", "b_" + typeBool.toString(), typeBool);
        Expr vi = prover.boundVarExpr("i", "i_" + typeInt.toString(), typeInt);
        Expr vr = prover.boundVarExpr("r", "r_" + typeReal.toString(), typeReal);
	Expr vref = prover.boundVarExpr("ref", "ref_" + typeRef.toString(), typeRef);
	/*
	Expr vf = prover.boundVarExpr("f", "f_" + typeField.toString(), typeField);
	Expr vfb = prover.boundVarExpr("fb", "fb_" + typeBoolField.toString(), typeBoolField);
	Expr vfi = prover.boundVarExpr("fi", "fi_" + typeIntField.toString(), typeIntField);
	Expr vfr = prover.boundVarExpr("fr", "fr_" + typeRealField.toString(), typeRealField);
	Expr vfref = prover.boundVarExpr("fref", "fref_" + typeRefField.toString(), typeRefField);
	*/
	setupCast(refToType, typeToRef, vref, vt);
        if (!optUseDatatype) {
            setupCast(valueToBool, boolToValue, vv, vb);
            setupCast(valueToInt, intToValue, vv, vi);
            setupCast(valueToReal, realToValue, vv, vr);
            setupCast(valueToRef, refToValue, vv, vref);
        }
	/*
	setupCast(fieldToBoolField, boolFieldToField, vf, vfb);
	setupCast(fieldToIntField, intFieldToField, vf, vfi);
	setupCast(fieldToRealField, realFieldToField, vf, vfr);
	setupCast(fieldToRefField, refFieldToField, vf, vfref);
	*/
    }

    // declare a fresh class classType(n),
    // using the datatype representation of JavaType
    protected Expr declareFreshClass(String name) throws cvc3.Cvc3Exception {
        assert (optUseDatatype);
        List args = new ArrayList();
        args.add(prover.ratExpr(classId++));
        Expr def = prover.datatypeConsExpr("classType", args);
        return prover.varExpr(name, typeType, def);
    }

    // special predefined symbols
    protected void setupSymbols() throws cvc3.Cvc3Exception {
	setupCast();

	// make sure that fClosedTime is applied to ref fields only,
	// not general fields, to avoid casting
	String nameFClosedTime = uniqueName(symFClosedTime);
	symbolMap.put(nameFClosedTime,
	  prover.createOp(nameFClosedTime, prover.funType(typeRefField, typeTime)));

        symbolMap.put(uniqueName(symRefValue), refToValue);

        // setup datatype constructors/selectors to avoid
        // their automatic creation as standard functions
        if (optUseDatatype) {
            Expr expr = null;
            List args = null;

            symbolMap.put("boolType", prover.datatypeConsExpr("boolType", new ArrayList()));
            symbolMap.put("charType", prover.datatypeConsExpr("charType", new ArrayList()));
            symbolMap.put("byteType", prover.datatypeConsExpr("byteType", new ArrayList()));
            symbolMap.put("shortType", prover.datatypeConsExpr("shortType", new ArrayList()));
            symbolMap.put("intType", prover.datatypeConsExpr("intType", new ArrayList()));
            symbolMap.put("longType", prover.datatypeConsExpr("longType", new ArrayList()));
            // :TODO: currently bigInt is defined as a synomym for long in the background predicate,
            // so we have to map it to the same constructor here
            symbolMap.put("bigIntType", prover.datatypeConsExpr("longType", new ArrayList()));
            symbolMap.put("floatType", prover.datatypeConsExpr("floatType", new ArrayList()));
            symbolMap.put("doubleType", prover.datatypeConsExpr("doubleType", new ArrayList()));
            symbolMap.put("voidType", prover.datatypeConsExpr("voidType", new ArrayList()));

            // hack to get classType constructor/selector
            args = new ArrayList();
            args.add(prover.ratExpr(0));
            expr = prover.datatypeConsExpr("classType", args);
            symbolMap.put(
              "classType",
              expr.getOp()
            );
            symbolMap.put(
              "classId",
              prover.datatypeSelExpr("classId", expr).getOp()
            );

            // hack to get arrayType constructor/selector
            args = new ArrayList();
            args.add(prover.datatypeConsExpr("boolType", new ArrayList()));
            expr = prover.datatypeConsExpr("arrayType", args);
            symbolMap.put(
              "arrayType",
              expr.getOp()
            );
            symbolMap.put(
              "elemType",
              prover.datatypeSelExpr("elemType", expr).getOp()
            );

            // for convenience, fix the class ids of common types
            symbolMap.put(uniqueName(symTObject), declareFreshClass(uniqueName(symTObject)));
            symbolMap.put(uniqueName(symTClass), declareFreshClass(uniqueName(symTClass)));
            symbolMap.put(uniqueName(symTException), declareFreshClass(uniqueName(symTException)));
            symbolMap.put(uniqueName(symTThrowable), declareFreshClass(uniqueName(symTThrowable)));
            symbolMap.put(uniqueName(symTCloneable), declareFreshClass(uniqueName(symTCloneable)));
            symbolMap.put(uniqueName(symTComparable), declareFreshClass(uniqueName(symTComparable)));
            symbolMap.put(uniqueName(symTSerializable), declareFreshClass(uniqueName(symTSerializable)));
            symbolMap.put(uniqueName(symTString), declareFreshClass(uniqueName(symTString)));
            symbolMap.put(uniqueName(symTCharSequence), declareFreshClass(uniqueName(symTCharSequence)));

            symbolMap.put(uniqueName(symIsClassType), isClassType);
            symbolMap.put(uniqueName(symIsArrayType), isArrayType);
        }

        if (optUseDatatype) {
            symbolMap.put(uniqueName(symIsBoolValue), isBoolValue);
            symbolMap.put(uniqueName(symIsIntValue), isIntValue);
            symbolMap.put(uniqueName(symIsRealValue), isRealValue);
            symbolMap.put(uniqueName(symIsRefValue), isRefValue);
        }

        // typeOf is applied to Ref only, so change its type appropriately
        // :UPDATE:
        // moved change  to definition in EscNodeBuilder to improve type inference)
	//String nameTypeOf = uniqueName(symTypeOf);
	//symbolMap.put(nameTypeOf,
	//  prover.createOp(nameTypeOf, prover.funType(typeRef, typeType)));
    }

    // setup predefined names to avoid name mangling by uniqueName
    protected void setupPredefinedNames() {
        // doesn't work: predefinedNames.put(symTType.name, "Type");
        // :TODO: what is this good for?
        // seems to be declared in vcs but then never used
        //predefinedNames.put("TYPE", "TYPE");
        // :TODO: is this ever used anywhere?
        //predefinedNames.put(symTArray.name, "arrayType");
        predefinedNames.put(symTBoolean.name, "boolType");
        predefinedNames.put(symTChar.name, "charType");
        predefinedNames.put(symTByte.name, "byteType");
        predefinedNames.put(symTShort.name, "shortType");
        predefinedNames.put(symTInt.name, "intType");
        predefinedNames.put(symTLong.name, "longType");
        predefinedNames.put(symTFloat.name, "floatType");
        predefinedNames.put(symTDouble.name, "doubleType");
        predefinedNames.put(symTBigInt.name, "bigIntType");
        predefinedNames.put(symTVoid.name, "voidType");
        predefinedNames.put(symClassType.name, "classType");
        predefinedNames.put(symRefValue.name, "javaRef");

        predefinedNames.put(symTObject.name, "T_java_lang_Object");
        predefinedNames.put(symTClass.name, "T_java_lang_Class");
        predefinedNames.put(symTException.name, "T_java_lang_Exception");
        predefinedNames.put(symTThrowable.name, "T_java_lang_Throwable");
        predefinedNames.put(symTCloneable.name, "T_java_lang_Cloneable");
        predefinedNames.put(symTComparable.name, "T_java_lang_Comparable");
        predefinedNames.put(symTSerializable.name, "T_java_io_Serializable");
        predefinedNames.put(symTString.name, "T_java_lang_String");
        predefinedNames.put(symTCharSequence.name, "T_java_lang_CharSequence");

        predefinedNames.put(symRefEQ.name, "refEQ");
        predefinedNames.put(symRefNE.name, "refNE");
        predefinedNames.put(symAllocLT.name, "allocLT");
        predefinedNames.put(symAllocLE.name, "allocLE");
        predefinedNames.put(symLockLT.name, "lockLT");
        predefinedNames.put(symLockLE.name, "lockLE");
        predefinedNames.put(symTypeEQ.name, "typeEQ");
        predefinedNames.put(symTypeNE.name, "typeNE");
        predefinedNames.put(symTypeLE.name, "isSubtype");
        predefinedNames.put(symIs.name, "is");
        predefinedNames.put(symCast.name, "cast");
        predefinedNames.put(symTypeOf.name, "typeOf");
        predefinedNames.put(symIsAllocated.name, "isAllocated");
        predefinedNames.put(symEClosedTime.name, "eClosedTime");
        predefinedNames.put(symFClosedTime.name, "fClosedTime");
        predefinedNames.put(symAsChild.name, "asChild");
        predefinedNames.put(symClassDown.name, "classDown");
        predefinedNames.put(symAsElems.name, "asElems");
        predefinedNames.put(symAsField.name, "asField");
        predefinedNames.put(symAsLockSet.name, "asLockSet");
        predefinedNames.put(symArrayLength.name, "arrayLength");
        predefinedNames.put(symArrayShapeOne.name, "arrayShapeOne");
        predefinedNames.put(symArrayShapeMore.name, "arrayShapeMore");
        predefinedNames.put(symArrayParent.name, "arrayParent");
        predefinedNames.put(symArrayPosition.name, "arrayPosition");
        predefinedNames.put(symArrayFresh.name, "arrayFresh");
        predefinedNames.put(symArray.name, "arrayType");
        predefinedNames.put(symUnset.name, "unset");
        predefinedNames.put(symIsNewArray.name, "isNewArray");
        predefinedNames.put(symIntern.name, "intern");
        predefinedNames.put(symInterned.name, "interned");
        predefinedNames.put(symStringCat.name, "stringCat");
        predefinedNames.put(symStringCatP.name, "stringCatP");
        predefinedNames.put(symNonNullElems.name, "nonnullelems");
        predefinedNames.put(symElemType.name, "elemType");
        predefinedNames.put(symMax.name, "max");
        predefinedNames.put(symArrayMake.name, "arrayMake");
        predefinedNames.put(symClassLiteral.name, "classLiteral");

        predefinedNames.put(symAlloc.name, "alloc");
        predefinedNames.put(symvAllocTime.name, "vAllocTime");
        predefinedNames.put(symLockPrec.name, "lockPrec");
        predefinedNames.put(symIntegralDiv.name, "integralDiv");
        predefinedNames.put(symIntegralMod.name, "integralMod");
        predefinedNames.put(symFloatingDiv.name, "floatingDiv");
        predefinedNames.put(symFloatingMod.name, "floatingMod");
        predefinedNames.put(symSL32.name, "intShiftL");
        predefinedNames.put(symSL64.name, "longShiftL");

        predefinedNames.put("getClass#", "getClass");
        //        predefinedNames.put("ecReturn", "ecReturn");
        //        predefinedNames.put("ecThrow", "ecThrow");
    }



    // escjava might use the same name with different types
    // in different queries using one prover instance
    // as cvc can't handle this, make the names unique by encoding their type
    public String uniqueName(FnSymbol fn) {
        // :TODO: is symTArray ever used?
        assert (fn != symTArray);

        String predefinedName = (String) predefinedNames.get(fn.name);
        if (predefinedName != null) {
            return predefinedName;
        }

	StringBuffer name = new StringBuffer(fn.name + ":");
	for (int i = 0; i < fn.argumentTypes.length; ++i) {
	    name.append("_" + fn.argumentTypes[i]);
	}
	name.append("_" + fn.retType);

	return toCvc3Input(name.toString());
	//return name.toString();
    }

    // :TODO: make readable as cvc textual input (helpful for debugging)
    public String toCvc3Input(String string) {
	String cvcSuitable = string;

	// :TODO: check if stuff we omitted when translating the background predicate
	// does actually occur in a test
	assert(!string.equals("intFirst"));
	assert(!string.equals("intLast"));
	assert(!string.equals("longFirst"));
	assert(!string.equals("longLast"));
	assert(!string.equals("arrayMake"));
	assert(!string.equals("arrayType"));
	assert(!string.equals("isNewArray"));
	// :TODO: test495 StackCheck.java
        assert(!string.equals("unset"));
	assert(!string.equals("stringCat"));
	assert(!string.equals("stringCatP"));
	assert(!string.equals("next"));
	assert(!string.equals("integralEQ"));
	assert(!string.equals("integralLE"));
	assert(!string.equals("integralLT"));
	assert(!string.equals("integralGE"));
	assert(!string.equals("integralGT"));
	assert(!string.equals("integralNE"));
	assert(!string.equals("integralAnd"));
	assert(!string.equals("integralOr"));
	assert(!string.equals("integralDiv"));
	assert(!string.equals("integralXor"));
	assert(!string.equals("integralMod"));
	assert(!string.equals("intShiftL"));
	assert(!string.equals("longShiftL"));
	assert(!string.equals("floatingEQ"));
	assert(!string.equals("floatingLE"));
	assert(!string.equals("floatingLT"));
	assert(!string.equals("floatingGE"));
	assert(!string.equals("floatingGT"));
	assert(!string.equals("floatingNE"));
	assert(!string.equals("floatingADD"));
	assert(!string.equals("floatingMUL"));
	assert(!string.equals("floatingMOD"));
	assert(!string.equals("refEQ"));
	assert(!string.equals("refNE"));

        // strictly speaking should first replace _ by __
	// to avoid non-unique renaming

	// not really sure what to do with unknown tags:
 	// "Unknown ESC tag _LT_251 (+56) _GT___type"
	// it's probably an error in the sorted translation.
        cvcSuitable = cvcSuitable.replaceAll(" ", "_");
        cvcSuitable = cvcSuitable.replaceAll("\\+", "_PL_");
        cvcSuitable = cvcSuitable.replaceAll("\\(", "_LP_");
        cvcSuitable = cvcSuitable.replaceAll("\\)", "_RP_");
        cvcSuitable = cvcSuitable.replaceAll("<:", "typeLE");
        cvcSuitable = cvcSuitable.replaceAll(":", "_");
        cvcSuitable = cvcSuitable.replaceAll(",", "_C_");
        cvcSuitable = cvcSuitable.replaceAll("=", "_EG_");
        cvcSuitable = cvcSuitable.replaceAll("<", "_LT_");
        cvcSuitable = cvcSuitable.replaceAll(">", "_GT_");
        cvcSuitable = cvcSuitable.replaceAll("@", "_AT_");
        cvcSuitable = cvcSuitable.replaceAll("#", "_SH_");
        cvcSuitable = cvcSuitable.replaceAll("!", "_EX_");
        cvcSuitable = cvcSuitable.replaceAll("%", "_PR_");
        cvcSuitable = cvcSuitable.replaceAll("-", "_MI_");
        cvcSuitable = cvcSuitable.replaceAll("\\|", "_PI_");
        cvcSuitable = cvcSuitable.replaceAll("\\.", "_D_");
        cvcSuitable = cvcSuitable.replaceAll("\\[", "_LB_");
        cvcSuitable = cvcSuitable.replaceAll("\\]", "_RB_");

        return cvcSuitable;
    }


    //
    // Mapping: Sort -> Type
    //

    // mapping of a basic sort (not function or predicate) to its cvc type
    protected Type mapValSort(Sort sort) throws Cvc3Exception {
	Cvc3Prover.print("mapValSort: " + sort);
	if (typeMap.containsKey(sort.name)) {
	    return (Type) typeMap.get(sort.name);
	} else if (sort.isSubSortOf(sortMap)) {
	    Sort fromSort = sort.getMapFrom();
	    Sort toSort = sort.getMapTo();
	    assert(fromSort != null);
	    assert(toSort != null);
	    Type type = prover.arrayType(mapValSort(fromSort), mapValSort(toSort));
	    typeMap.put(sort.name, type);
	    return type;
	} else {
	    ErrorSet.fatal("mapValSort");
	    assert(false);
	    return prover.createType("UnknownType");
	}
    }

    // mapping of a function symbol to its cvc type
    protected Type mapFunSort(FnSymbol symbol) throws Cvc3Exception {
	Cvc3Prover.print("mapFunSort: " + symbol);
	if (typeMap.containsKey(symbol.name)) {
	    return (Type) typeMap.get(symbol.name);
	} else {
	    List argumentTypes = new ArrayList();
	    for (int i = 0; i < symbol.argumentTypes.length; ++i) {
		argumentTypes.add(mapValSort(symbol.argumentTypes[i]));
	    }
	    Type returnType = mapValSort(symbol.retType);
	    Type type = prover.funType(argumentTypes, returnType);
	    typeMap.put(symbol.name, type);
	    return type;
	}
    }

    // mapping of a predicate symbol to its cvc type
    protected Type mapPredSort(PredSymbol symbol) throws Cvc3Exception {
	return mapFunSort(symbol);
    }



    //
    // Mapping: Symbol -> Constant/Operator
    //

    // mapping of a constant symbol to its cvc constant expression
    protected Expr mapValSymbol(FnSymbol symbol) throws Cvc3Exception {
	Cvc3Prover.print("mapValSymbol: " + symbol.name);
	assert(symbol.argumentTypes.length == 0);
	String name = uniqueName(symbol);

	if (symbolMap.containsKey(name)) {
	    return (Expr) symbolMap.get(name);
	} else {
            Type type = mapValSort(symbol.retType);
	    Expr var = null;
            if (optUseDatatype && type.equals(typeType)) {
                /*
                if (!name.startsWith("T_")) {
                    System.out.println(" mapJavaType: UNEXPECTED TYPE NAME: " + name);
                }
                */
                // datatype: make this a unique class,
                // thus ensuring it being distinct from all other types
                var = declareFreshClass(name);
            } else {
                // just make it a JavaType
                var = prover.varExpr(name, type);
            }
	    symbolMap.put(name, var);
	    return var;
	}
    }

    // mapping of a function symbol to its cvc operator
    protected Op mapFunSymbol(FnSymbol symbol) throws Cvc3Exception {
	Cvc3Prover.print("mapFunSymbol: " + symbol.name);
	String name = uniqueName(symbol);
	if (symbolMap.containsKey(name)) {
	    return (Op) symbolMap.get(name);
	} else {
            Op op;

            // drop classLiteral (and implicitly replace it by typeToRef),
            // for it's doing nothing but providing a hint for triggers,
            // which doesn't seem to be needed in practice
            if (!optNonnullelements && symbol == symNonNullElems) {
                //(DEFPRED (nonnullelements x e)
                //   (AND (NEQ x null)
                //	(FORALL (i)
                //	   (IMPLIES (AND (<= 0 i)
                //			 (< i (arrayLength x)))
                //		    (NEQ (select (select e x) i) null)))))
                Expr x = prover.boundVarExpr("x", "x_" + typeRef.toString(), typeRef);
                Expr e = prover.boundVarExpr("elems", "elems_" + typeElems.toString(), typeElems);
                Expr i = prover.boundVarExpr("i", "i_" + typeInt.toString(), typeInt);
                Expr t0 = prover.funExpr(mapFunSymbol(symArrayLength), x);
                Expr t1 = prover.readExpr(e, x);
                Expr t2 = prover.readExpr(t1, i);
                Expr p0 = prover.leExpr(prover.ratExpr(0), i);
                Expr p1 = prover.ltExpr(i, t0);
                Expr p2 = prover.notExpr(prover.eqExpr(t2, cast(escNull.getExpr(), typeValue)));
                Expr p3 = prover.andExpr (p0, p1);
                Expr p4 = prover.impliesExpr(p3, p2);
                Expr p5 = prover.forallExpr(Arrays.asList (new Expr[] { i } ), p4);
                Expr p6 = prover.notExpr(prover.eqExpr(x, escNull.getExpr()));
                Expr p7 = prover.andExpr(p6, p5);

                Op lambda = prover.lambdaExpr(Arrays.asList(new Expr[] { x, e }), p7);
                op = prover.createOp(name, mapFunSort(symbol), lambda.getExpr());
            }

            else if (false && !optIsallocated && symbol == symIsAllocated) {
                //(DEFPRED (isAllocated x a0) (< (vAllocTime x) a0))
                Expr x = prover.boundVarExpr("x", "x_" + typeRef.toString(), typeRef);
                Expr a0 = prover.boundVarExpr("a0", "a0_" + typeTime.toString(), typeTime);
                Expr t0 = prover.funExpr(mapFunSymbol(symvAllocTime), x);
                Expr p0 = prover.ltExpr(t0, a0);
                Op lambda = prover.lambdaExpr(Arrays.asList(new Expr[] { x, a0 }), p0);
                op = prover.createOp(name, mapFunSort(symbol), lambda.getExpr());
            }

            // drop classLiteral (and implicitly replace it by typeToRef),
            // for it's doing nothing but providing a hint for triggers,
            // which doesn't seem to be needed in practice
            else if (!optUseClassLiteral && symbol == symClassLiteral) {
                /* classLiteral as synonym of typeToRef not effective in patterns
                Expr v = prover.boundVarExpr("v", "v_" + typeType.toString(), typeType);
                List vars = Arrays.asList(new Expr[] { v });
                Expr def = prover.funExpr(typeToRef, v);
                Op lambda = prover.lambdaExpr(vars, def);
                op = prover.createOp(name, mapFunSort(symbol), lambda.getExpr());
                */
                op = typeToRef;
            }

            else {
                op = prover.createOp(name, mapFunSort(symbol));
            }
            
            // special treatment for builtin transitivity
            if (optBuiltinTrans && symbol == symTypeLE) {
                op = prover.transClosure(op);
            }

	    symbolMap.put(name, op);
	    return op;
	}
    }

    // mapping of a predicate symbol to its cvc operator
    protected Op mapPredSymbol(PredSymbol symbol) throws Cvc3Exception {
	return mapFunSymbol(symbol);
    }


    //
    // Mapping: Bound Variables
    //

    // mapping of a bound variable to its cvc bound variable
    protected Cvc3Any getBoundVarExpr(QuantVar bvar) {
	String key = bvar.name + ":" + bvar.type;
	if (varsMap.containsKey(key)) {
	    return (Cvc3Any) varsMap.get(key);
	} else {
	    try {
		Expr e = prover.boundVarExpr(toCvc3Input(bvar.name),
					     toCvc3Input(bvar.name + bvar.type.toString()),
					     mapValSort(bvar.type));
		Cvc3Any var = (Cvc3Any) wrapExpr(e);
		//Cvc3Any var = wrapAnyExpr(e, bvar.type);
		varsMap.put(key, var);
		return var;
	    } catch (cvc3.Cvc3Exception e) {
		ErrorSet.fatal(e.toString());
		throw new Error(e);
	    }
	}
    }


    // cvc predicates for memoization, with rewriting for atoms
    public Cvc3Pred registerCvc3Pred(Object key, Cvc3Pred p) throws Cvc3Exception {
	Expr e = p.getExpr();
	assert(e.getType().equals(prover.boolType()));
	assert(!(p instanceof Cvc3Label)); // should not register labels
	if (rewriteCvc3 && e.isPropAtom()) {
	    if (prover.simplify(e).equals(prover.trueExpr())) {
		Cvc3Pred node = transferLabels(p, (Cvc3Pred) buildTrue());
		assert(!escMap.containsKey(key));
		escMap.put(key, node);
		exprMap.put(node.getExpr(), node);
		assert(escMap.containsKey(key));
		return node;
	    } else if (prover.simplify(e).equals(prover.falseExpr())) {
		Cvc3Pred node = transferLabels(p, (Cvc3Pred) buildFalse());
		assert(!escMap.containsKey(key));
		escMap.put(key, node);
		exprMap.put(node.getExpr(), node);
		return node;
	    }
	}
	// no simplification
	escMap.put(key, p);
        exprMap.put(p.getExpr(), p);
	return p;
    }


    //
    // type conversions
    //

    // find common subtype of t1 and t2,
    // assuming that one is an instance of the other.
    public Type unify(Type t1, Type t2) {
	if (t1.equals(t2)) {
	    return t1;
	}

	// int to real
	else if (t1.equals(typeInt) && t2.equals(typeReal)) {
	    return typeReal;
	} else if (t1.equals(typeReal) && t2.equals(typeInt)) {
	    return typeReal;
	}

	// value to bool/int/real/ref
	else if (t1.equals(typeValue)) {
	    if (t2.equals(typeBool)
		|| 
		t2.equals(typeInt)
		||
		t2.equals(typeReal)
		||
		t2.equals(typeRef)) {
		return t2;
	    }
	}
	else if (t2.equals(typeValue)) {
	    if (t1.equals(typeBool)
		|| 
		t1.equals(typeInt)
		||
		t1.equals(typeReal)
		||
		t1.equals(typeRef)) {
		return t1;
	    }
	}

	// fields to field instances
	else if (t1.equals(typeField)) {
	    if (t2.equals(typeBoolField)
		||
		t2.equals(typeIntField)
		||
		t2.equals(typeRealField)
		||
		t2.equals(typeRefField)) {
		return t2;
	    }
	}
	else if (t2.equals(typeField)) {
	    if (t1.equals(typeBoolField)
		||
		t1.equals(typeIntField)
		||
		t1.equals(typeRealField)
		||
		t1.equals(typeRefField)) {
		return t1;
	    }
	}

	ErrorSet.fatal("unify: " + t1 + " <---> " + t2);
	assert(false);
	return typeError;
    }


    //
    // cast an expression to the given type
    //

    public boolean isArithType(Expr expr) throws Cvc3Exception {
	Type t = expr.getChild(0).getType();
	return t.equals(typeInt) || t.equals(typeReal);
    }

    public boolean isCastOp(Op op) {
	for (int i = 0; i < casts.length; ++i) {
	    if (casts[i].equals(op)) {
		return true;
	    }
	}
	return false;
    }

    // expr should be cast to cast(expr),
    // but if expr has the form inverse(expr),
    // with inverse the inverse of case,
    // then cast(inverse(expr)) simplifies to expr
    public Expr simplifyCast(Expr expr, Op cast, Op inverse, Op inverse2) throws Cvc3Exception {
	if (expr.isApply()
	    &&
	    (expr.getOp().equals(inverse) || expr.getOp().equals(inverse2))) {
	    assert(expr.arity() == 1);
	    return expr.getChild(0);
	} else {
	    return prover.funExpr(cast, expr);
	}
    }

    public Expr simplifyCast(Expr expr, Op cast, Op inverse) throws Cvc3Exception {
	return simplifyCast(expr, cast, inverse, null);
    }

    public boolean sameCast(Expr expr1, Expr expr2) throws Cvc3Exception {
	if (expr1.isApply() && expr2.isApply()) {
	    Op op1 = expr1.getOp();
	    Op op2 = expr2.getOp();
	    if (op1.equals(op2) && isCastOp(op1) && isCastOp(op2)) {
		assert(expr1.arity() == 1);
		assert(expr2.arity() == 1);
		return true;
	    }
	}

	return false;
    }


    public Expr cast(Expr expr, Sort sort) throws Cvc3Exception {
	return cast(expr, mapValSort(sort));
    }

    public Expr cast(Expr expr, Type type) throws Cvc3Exception {
	Cvc3Prover.print("cast: " + expr + " <---> " + type);
	Type eType = expr.getType();
	if (eType.equals(type)) {
	    return expr;
	}

	// real <-> int
	// :TODO: cvc doesn't care if it's integer or real ?
	// needed as Lifter distinguishes between Cvc3Int and Cvc3Real
	else if (type.equals(typeInt) && eType.equals(typeReal)) {
	    return expr;
	    //return prover.funExpr(realToInt, expr);
	} else if (type.equals(typeReal) && eType.equals(typeInt)) {
	    //return prover.funExpr(intToReal, expr);
	    return expr;
	}

	// ref <-> type
	// e.g needed in: (NEQ (|getClass#| |brokenObj<8>|) null)
	else if (type.equals(typeRef) && eType.equals(typeType)) {
	    return simplifyCast(expr, typeToRef, refToType);
	    //return prover.funExpr(typeToRef, expr);
	} else if (type.equals(typeType) && eType.equals(typeRef)) {
	    return simplifyCast(expr, refToType, typeToRef);
	    //return prover.funExpr(refToType, expr);
	}

	// bool <-> pred
	// can we get away doing nothing?
	// cvc doesn't really care if a int is used as a real,
	// or a real which is actually an int is used as an int.
	else if (type.equals(typePred) && eType.equals(typeBool)) {
	    assert(false);
	    return prover.eqExpr(expr, ((Cvc3Bool)buildBool(true)).getExpr());
	} else if (type.equals(typeBool) && eType.equals(typePred)) {
	    assert(false);
	    return prover.iteExpr(expr, ((Cvc3Bool)buildBool(true)).getExpr(), ((Cvc3Bool)buildBool(false)).getExpr());
	}

	// bool/int/real/ref to value
	else if (type.equals(typeValue)) {
	    if (eType.equals(typeBool)) {
		return simplifyCast(expr, boolToValue, valueToBool);
		//return prover.funExpr(boolToValue, expr);
	    } else if (eType.equals(typeInt)) {
		return simplifyCast(expr, intToValue, valueToInt, valueToReal);
		/*
		if (expr.isApply() && expr.getOp().equals(valueToInt)) {
		    System.out.println (valueToInt);
		    assert(expr.arity() == 1);
		    return expr.getChild(0);
			//if (op.getExpr().equals(valueToInt.getExpr())) {
			//System.out.println ("intToValue of: " + expr);
			//System.out.println (expr.arity());
			//System.out.println (expr.getChild(0));
		}
		return prover.funExpr(intToValue, expr);
		*/
	    } else if (eType.equals(typeReal)) {
		return simplifyCast(expr, realToValue, valueToReal, valueToReal);
		//return prover.funExpr(realToValue, expr);
	    } else if (eType.equals(typeRef)) {
		return simplifyCast(expr, refToValue, valueToRef);
		//return prover.funExpr(refToValue, expr);
	    }
	}
	// value to bool/int/real/ref
	else if (type.equals(typeBool)) {
	    if (eType.equals(typeValue)) {
		return simplifyCast(expr, valueToBool, boolToValue);
		//return prover.funExpr(valueToBool, expr);
	    }
	} else if (type.equals(typeInt)) {
	    if (eType.equals(typeValue)) {
		return simplifyCast(expr, valueToInt, intToValue, realToValue);
		//return prover.funExpr(valueToInt, expr);
	    }
	} else if (type.equals(typeReal)) {
	    if (eType.equals(typeValue)) {
		return simplifyCast(expr, valueToReal, realToValue, intToValue);
		//return prover.funExpr(valueToReal, expr);
	    }
	} else if (type.equals(typeRef)) {
	    if (eType.equals(typeValue)) {
		return simplifyCast(expr, valueToRef, refToValue);
		//return prover.funExpr(valueToRef, expr);
	    }	    
	}
	/*
	// bool/int/real/ref field instances to field
	else if (type.equals(typeField)) {
	    if (eType.equals(typeBoolField)) {
		return simplifyCast(expr, boolFieldToField, fieldToBoolField);
		//return prover.funExpr(boolFieldToField, expr);
	    } else if (eType.equals(typeIntField)) {
		return simplifyCast(expr, intFieldToField, fieldToIntField);
		//return prover.funExpr(intFieldToField, expr);
	    } else if (eType.equals(typeRealField)) {
		return simplifyCast(expr, realFieldToField, fieldToRealField);
		//return prover.funExpr(realFieldToField, expr);
	    } else if (eType.equals(typeRefField)) {
		return simplifyCast(expr, refFieldToField, fieldToRefField);
		//return prover.funExpr(refFieldToField, expr);
	    }
	}
	// field to bool/int/real/ref field instances
	else if (type.equals(typeBoolField)) {
	    if (eType.equals(typeField)) {
		return simplifyCast(expr, fieldToBoolField, boolFieldToField);
		//return prover.funExpr(fieldToBoolField, expr);
	    }
	} else if (type.equals(typeIntField)) {
	    if (eType.equals(typeField)) {
		return simplifyCast(expr, fieldToIntField, intFieldToField);
		//return prover.funExpr(fieldToIntField, expr);
	    }
	} else if (type.equals(typeRealField)) {
	    if (eType.equals(typeField)) {
		return simplifyCast(expr, fieldToRealField, realFieldToField);
		//return prover.funExpr(fieldToRealField, expr);
	    }
	} else if (type.equals(typeRefField)) {
	    if (eType.equals(typeField)) {
		return simplifyCast(expr, fieldToRefField, refFieldToField);
		//return prover.funExpr(fieldToRefField, expr);
	    }
	}
	*/
	ErrorSet.fatal("cast: " + eType + " <---> " + type);
	assert(false);
	return expr;
    }


    //
    // wrap an expression in an esc node corresponding to the given sort
    //
    // this doesn't mean that the contained expression is of the corresponding type,
    // just that it is convertible.
    // E.g. a Cvc3Int can contain an expression of type typeReal.
    //


    Cvc3Term wrapExpr(Expr e) throws Cvc3Exception {
	return wrapExpr(e, e.getType());
    }

    Cvc3Term wrapExpr(Expr e, Sort sort) throws Cvc3Exception {
	return wrapExpr(e, mapValSort(sort));
    }

    Cvc3Term wrapExpr(Expr e, Type type) throws Cvc3Exception {
	if (type.equals(typePred)) {
	    return new Cvc3Pred(e);
	} else if (type.equals(typeValue)) {
	    return new Cvc3Value(e);
	} else if (type.equals(typeBool)) {
	    return new Cvc3Bool(e);
	} else if (type.equals(typeInt)) {
	    return new Cvc3Int(e);
	} else if (type.equals(typeReal)) {
	    return new Cvc3Real(e);
	} else if (type.equals(typeRef)) {
	    return new Cvc3Ref(e);
	} else if (type.equals(typeString)) {
	    assert(sortString.isSubSortOf(sortRef));
	    return new Cvc3Ref(e);

	} else if (type.equals(typeShape)) {
	    assert(sortShape.isSubSortOf(sortAny));
	    return new Cvc3Any(e);
	} else if (type.equals(typeTime)) {
	    assert(sortTime.isSubSortOf(sortInt));
	    return new Cvc3Int(e);
	} else if (type.equals(typeType)) {
	    assert(sortType.isSubSortOf(sortRef));
	    return new Cvc3Ref(e);
	} else if (type.equals(typeLock)) {
	    assert(sortType.isSubSortOf(sortRef));
	    return new Cvc3Ref(e);
	    
	} else if (type.equals(typeField)
		   ||
		   type.equals(typeBoolField)
		   ||
		   type.equals(typeIntField)
		   ||
		   type.equals(typeRealField)
		   ||
		   type.equals(typeRefField)
		   ||
		   type.equals(typeArray)
		   ||
		   type.equals(typeElems)
		   ||
		   type.equals(typeLockSet)) {
	    return new Cvc3Map(e);
	} else {
	    System.out.println("wrapexpr failed");
	    System.out.println(e);
	    System.out.println(type);
	    assert (false);
	    return new Cvc3Term();
	}
    }



    
    //
    // NodeBuilder interface
    //

    protected String integralPrintName(long n) {
	Cvc3Prover.print("!integralPrintName");
	assert(false);
	return String.valueOf(n);
    }

    public SAny buildFnCall(FnSymbol fn, SAny[] args) {
	Cvc3Prover.print("!buildFnCall: " + fn);
	for (int i = 0; i < args.length; ++i) {
	    Cvc3Prover.print("  " + args[i]);
	}
	assert(fn.retType != sortPred);
	// :TODO: check if stuff we omitted when translating the background predicate
	assert(fn != symArrayMake);
	assert(fn != symIsNewArray);
	/* Lifter public vars, not static though
	assert(fn != symValueToBool);
	assert(fn != symValueToInt);
	assert(fn != symValueToReal);
	assert(fn != symValueToRef);
	assert(fn != symValueToPred);
	assert(fn != symBoolToValue);
	assert(fn != symIntToValue);
	assert(fn != symRealToValue);
	assert(fn != symRefToValue);
	assert(fn != symIntToReal);
	assert(fn != symRealToInt);
	assert(fn != symPredToBool); */
	assert(fn != symNonNullElems);
	assert(!fn.name.equals("intFirst"));
	assert(!fn.name.equals("intLast"));
	assert(!fn.name.equals("longFirst"));
	assert(!fn.name.equals("longLast"));
	assert(!fn.name.equals("arrayMake"));
	assert(!fn.name.equals("arrayType"));
	// :TODO: test495 StackCheck.java assert(!fn.name.equals("unset"));
	//assert(!fn.name.equals("stringCat"));
	//assert(!fn.name.equals("stringCatP"));
	assert(!fn.name.equals("next"));
	assert(!fn.name.equals("integralEQ"));
	assert(!fn.name.equals("integralLE"));
	assert(!fn.name.equals("integralLT"));
	assert(!fn.name.equals("integralGE"));
	assert(!fn.name.equals("integralGT"));
	assert(!fn.name.equals("integralNE"));
	assert(!fn.name.equals("integralAnd"));
	assert(!fn.name.equals("integralOr"));
	assert(!fn.name.equals("integralDiv"));
	assert(!fn.name.equals("integralXor"));
	assert(!fn.name.equals("integralMod"));
	assert(!fn.name.equals("intShiftL"));
	assert(!fn.name.equals("longShiftL"));
	assert(!fn.name.equals("floatingEQ"));
	assert(!fn.name.equals("floatingLE"));
	assert(!fn.name.equals("floatingLT"));
	assert(!fn.name.equals("floatingGE"));
	assert(!fn.name.equals("floatingGT"));
	assert(!fn.name.equals("floatingNE"));
	assert(!fn.name.equals("floatingADD"));
	assert(!fn.name.equals("floatingMUL"));
	assert(!fn.name.equals("floatingMOD"));
	assert(!fn.name.equals("refEQ"));
	assert(!fn.name.equals("refNE"));
	//	assert(!fn.name.equals(symAsChild.name));
	try {
	    List key = Arrays.asList(new Object[] { "buildFnCall", uniqueName(fn), args });

	    Expr expr;
	    //Type rType;
	    if (escMap.containsKey(key)) {
		return (Cvc3Any) escMap.get(key);
	    } else if (args.length == 0) {
		expr = mapValSymbol(fn);
		//rType = expr.getType();
	    } else {
		Op f = mapFunSymbol(fn);
		if (fn == symAsField) {
		    //System.out.println("ASFIELD");
		    //assert (args.length == 2);
		    Type fieldType = ((Cvc3Any)args[0]).getExpr().getType();
		    if (fieldType.equals(typeBoolField)) {
			f = asFieldBool;
		    } else if (fieldType.equals(typeIntField)) {
			f = asFieldInt;
		    } else if (fieldType.equals(typeRealField)) {
			f = asFieldReal;
		    } else if (fieldType.equals(typeRefField)) {
			f = asFieldRef;
		    } else {
			System.out.println(args[0]);
			System.out.println(fieldType);
			assert(false);
		    }
		    /*		} else if (fn == symFClosedTime) {
		    System.out.println("FCLOSEDTIME");
		    assert (args.length == 1);
		    Type fieldType = ((Cvc3Any)args[0]).getExpr().getType();
		    if (fieldType.equals(typeBoolField)) {
			f = fClosedTimeBool;
		    } else if (fieldType.equals(typeIntField)) {
			f = fClosedTimeInt;
		    } else if (fieldType.equals(typeRealField)) {
			f = fClosedTimeReal;
		    } else if (fieldType.equals(typeRefField)) {
			f = fClosedTimeRef;
		    } else {
			System.out.println(args[0]);
			System.out.println(fieldType);
			assert(false);
			}*/
		} else {
		    f = mapFunSymbol(fn);
		}
		Type fType = f.getExpr().getType();
		assert(fType.isFunction());
		assert(fType.arity() == args.length + 1);
		List children = new ArrayList();//cast((Cvc3Any[])args, fType);
		for (int i = 0; i < args.length; ++i) {
		    //assert(escMap.containsKey(args[i]));
		    Expr e = cast(((Cvc3Any)args[i]).getExpr(), fType.getChild(i));
		    children.add(e);
		}
	
		expr = prover.funExpr(f, children);
	    }
	    
	    Cvc3Any node = (Cvc3Any) wrapExpr(expr, fn.retType);
	    escMap.put(key, node);
            exprMap.put(node.getExpr(), node);
	    return node;

	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }

    public SAny buildConstantRef(FnSymbol c) {
	Cvc3Prover.print("!buildConstantRef: " + c);
	return new Cvc3Any();
    }

    public SAny buildQVarRef(QuantVar v) {
	Cvc3Prover.print("!buildQVarRef: " + v);
	return getBoundVarExpr(v);
    }

    //@ requires \nonnullelements(args);
    public SPred buildPredCall(PredSymbol fn, SAny[] args) {
	Cvc3Prover.print("!buildPredCall: " + fn + " " + args);
	for (int i = 0; i < args.length; ++i) {
	    Cvc3Prover.print("  " + args[i]);
	}
	assert(fn.retType == sortPred);

	// these are used to convert predicates to functions of type bool,
	// where cvc uses boolToPred / predToBool instead
	assert(!fn.name.equals("boolAnd"));
	assert(!fn.name.equals("boolEq"));
	assert(!fn.name.equals("boolImplies"));
	assert(!fn.name.equals("boolNE"));
	assert(!fn.name.equals("boolNot"));
	assert(!fn.name.equals("boolOr"));

	// symRefEQ is rewritten to =
	if (fn == symRefEQ) {
	    assert(args.length == 2);
	    return buildAnyEQ(args[0], args[1]);
	}
	if (fn == symRefNE) {
	    assert(args.length == 2);
	    return buildAnyNE(args[0], args[1]);
	}

	// symTypeEq is rewritten to =
	if (fn == symTypeEQ) {
	    assert(args.length == 2);
	    return buildAnyEQ(args[0], args[1]);
	}
	if (fn == symTypeNE) {
	    assert(args.length == 2);
	    return buildAnyNE(args[0], args[1]);
	}

	//(DEFPRED (isAllocated x a0) (< (vAllocTime x) a0))
	if (optIsallocated && fn == symIsAllocated) {
	    assert(args.length == 2);
	    SAny t = buildFnCall(symvAllocTime, new SAny[] { args[0] });
	    return buildArithPred(predLT, (SValue)t, (SValue)args[1]);
        }

	if (fn == symAllocLE) {
            //assert (false);
	    //assert(args.length == 2);
	    //return buildArithPred(
	    //  predLE,
	    //  buildValueConversion(sortRef, sortValue, (SValue)args[0]),
	    //  buildValueConversion(sortRef, sortValue, (SValue)args[1]));
	    return buildArithPred(predLE, (SValue)args[0], (SValue)args[1]);
	}
	if (fn == symAllocLT) {
            //            assert (false);
	    //assert(args.length == 2);
	    //return buildArithPred(
	    //  predLT,
	    //  buildValueConversion(sortRef, sortValue, (SValue)args[0]),
	    //  buildValueConversion(sortRef, sortValue, (SValue)args[1]));
	    return buildArithPred(predLT, (SValue)args[0], (SValue)args[1]);
	}

	//:TODO: replacing orderings over locks (i.e. sortRef) by
	// arithmetic ordering, needs cast: ref -> value -> int
	// perhaps just axiomatizing the DEFPRED's might be faster?
	//(DEFPRED (lockLE x y) (<= x y))
	if (fn == symLockLE) {
	    assert(args.length == 2);
	    SReal t0 = (SReal) buildFnCall(symLockPrec, new SAny[] { args[0] });
	    SReal t1 = (SReal) buildFnCall(symLockPrec, new SAny[] { args[1] });
	    return buildArithPred(predLE, t0, t1);
            //buildValueConversion(sortRef, sortValue, (SValue)args[0]),
	    //buildValueConversion(sortRef, sortValue, (SValue)args[1]));
	}
	//(DEFPRED (lockLT x y) (< x y))
	if (fn == symLockLT) {
	    assert(args.length == 2);
	    SReal t0 = (SReal) buildFnCall(symLockPrec, new SAny[] { args[0] });
	    SReal t1 = (SReal) buildFnCall(symLockPrec, new SAny[] { args[1] });
	    //buildValueConversion(sortRef, sortValue, (SValue)args[0]),
            //buildValueConversion(sortRef, sortValue, (SValue)args[1]));
	    return buildArithPred(predLT, t0, t1);
	}
	//(DEFPRED (nonnullelements x e)
	//   (AND (NEQ x null)
	//	(FORALL (i)
	//	   (IMPLIES (AND (<= 0 i)
	//			 (< i (arrayLength x)))
	//		    (NEQ (select (select e x) i) null)))))
	if (optNonnullelements && fn == symNonNullElems) {
	    assert(args.length == 2);
	    SValue x = (SValue) args[0];
	    SValue e = (SValue) args[1];
	    QuantVar vi = registerQuantifiedVariable("i", sortInt);
	    SAny t0 = buildFnCall(symArrayLength, new SAny[] { x });
	    SAny t1 = buildSelect((SMap)e, x);
	    SAny t2 = buildSelect((SMap)t1, (SInt)buildQVarRef(vi));
	    SPred p0 = buildAnyNE(x, buildNull());
	    SPred p1 = buildIntPred(predLE, buildInt(0), (SInt)buildQVarRef(vi));
	    SPred p2 = buildIntPred(predLT, (SInt)buildQVarRef(vi), (SInt)t0);
	    SPred p3 = buildAnyNE(t2, buildNull());
	    SPred p4 = buildAnd(new SPred[] { p1, p2 } );
	    SPred p5 = buildImplies(p4, p3);
            if (optUseDatatype) {
                p5 = buildImplies(buildPredCall(symIsRefValue, new SAny[] { t2 }), p5);
            }
	    QuantVar[] vars0 = new QuantVar[] { vi };
	    SPred p6 = buildForAll(vars0, p5, null, null);
	    SPred p7 = buildAnd(new SPred[] { p0, p6 } );
	    return p7;
	}
	try {
	    List key = Arrays.asList(new Object[] { "buildPredCall", uniqueName(fn), args });

	    Expr expr;
	    if (escMap.containsKey(key)) {
		return (Cvc3Pred) escMap.get(key);
	    } else if (args.length == 0) {
		expr = mapValSymbol(fn);
		//rType = expr.getType();
	    } else {
		Op f = mapPredSymbol(fn);
		Type fType = f.getExpr().getType();
		assert(fType.arity() == args.length + 1);
		List children = new ArrayList();
		for (int i = 0; i < args.length; ++i) {
		    //assert(escMap.containsKey(args[i]));
		    Expr e = cast(((Cvc3Term)args[i]).getExpr(), fType.getChild(i));
		    children.add(e);
		}
		
		expr = prover.funExpr(f, children);
	    }
	    Cvc3Pred node = new Cvc3Pred(expr);
	    return registerCvc3Pred(key, node);

	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }
	

    // Boolean connectives

    public SPred buildImplies(SPred _arg1, SPred _arg2) {
	Cvc3Prover.print("!buildImplies: " + _arg1 + " " + _arg2);
	return buildOr(buildNot(_arg1), _arg2);
    }

    public SPred buildIff(SPred _arg1, SPred _arg2) {
	Cvc3Prover.print("!buildIff: " + _arg1 + " " + _arg2);
	return buildAnd(buildImplies(_arg1, _arg2), buildImplies(_arg2, _arg1));
    }

    public SPred buildXor(SPred _arg1, SPred _arg2) {
	Cvc3Prover.print("!buildXor: " + _arg1 + " " + _arg2);
	return buildNot(buildIff(_arg1, _arg2));
    }

    public SPred buildAnd(SPred _arg1, SPred _arg2) {
	SPred[] args = { _arg1, _arg2 };
	return buildAnd(args);
    }

    public SPred buildAnd(SPred[] args) {
	Cvc3Prover.print("!buildAnd: " + args);
	for (int i = 0; i < args.length; ++i) {
	    Cvc3Prover.print("  " + args[i]);
	}
	try {
	    List key = Arrays.asList(new Object[] { "buildAnd", args });
	    if (escMap.containsKey(key)) {
		return (Cvc3Pred) escMap.get(key);
	    } else {
		Cvc3Pred node = null;
		boolean rewritten = false;
		if (rewrite) {
		    for (int i = 0; !rewritten && i < args.length; ++i) {
			Expr ci = ((Cvc3Pred)args[i]).getExpr();
			if (ci.equals(prover.falseExpr())
			    //||
			    //(rewriteCvc3 && prover.simplify(ci).equals(prover.falseExpr()))
			    ) {
			    rewritten = true;
			    node = transferLabels((Cvc3Pred)args[i], (Cvc3Pred)buildFalse());
			}
		    }

		    // p and -p
		    for (int i = 0; !rewritten && i < args.length - 1; ++i) {
			Expr ci = ((Cvc3Pred)args[i]).getExpr();
			for (int j = 0; !rewritten && i < args.length; ++i) {
			    Expr cj = ((Cvc3Pred)args[i]).getExpr();
			    if (ci.equals(prover.notExpr(cj)) || cj.equals(prover.notExpr(ci))) {
				rewritten = true;
				node = transferLabels((Cvc3Pred)args[i], (Cvc3Pred)buildFalse());
				node = transferLabels((Cvc3Pred)args[j], node);
			    }
			}
		    }
		}
		if (!rewritten) {
		    List children = new ArrayList();
		    List childrenExpr = new ArrayList();
		    //:TODO: omitted needs to be moved to left child, not to parent node
		    List omitted = new ArrayList();
		    for (int i = 0; i < args.length; ++i) {
			Expr ci = ((Cvc3Pred)args[i]).getExpr();
			if (!ci.equals(prover.trueExpr())
			    //||
			    //(rewriteCvc3 && prover.simplify(ci).equals(prover.trueExpr()))
			    ) {
			    children.add(args[i]);
			    childrenExpr.add(((Cvc3Pred)args[i]).getExpr());
			} else {
			    omitted.add(args[i]);
			}
		    }
		    if (children.size() == 0) {
			node = (Cvc3Pred) buildTrue();
		    } else if (children.size() == 1) {
			node = (Cvc3Pred) children.get(0);
		    } else {
			node = new Cvc3Pred(prover.andExpr(childrenExpr));
			Iterator i = children.iterator();
			while (i.hasNext()) {
			    node.addChild((Cvc3Pred)i.next());
			}
			assert(node.getExpr().arity() == node.getChildren().size());
			assert(args.length >= node.getExpr().arity());
		    }
		    //node = transferLabels(omitted, node);
		}
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }

    public SPred buildOr(SPred _arg1, SPred _arg2) {
	SPred[] args = { _arg1, _arg2 };
	return buildOr(args);
    }

    public SPred buildOr(SPred[] args) {
	Cvc3Prover.print("!buildOr: " + args);
	for (int i = 0; i < args.length; ++i) {
	    Cvc3Prover.print("  " + args[i]);
	}
	try {
	    List key = Arrays.asList(new Object[] { "buildOr", args });
	    if (escMap.containsKey(key)) {
		return (Cvc3Pred) escMap.get(key);
	    } else {
		Cvc3Pred node = null;
		boolean rewritten = false;
		if (rewrite) {
		    // true or p
		    for (int i = 0; !rewritten && i < args.length; ++i) {
			Expr ci = ((Cvc3Pred)args[i]).getExpr();
			if (ci.equals(prover.trueExpr())
			    //||
			    //(rewriteCvc3 && prover.simplify(ci).equals(prover.trueExpr()))
			    ) {
			    rewritten = true;
			    node = transferLabels((Cvc3Pred)args[i], (Cvc3Pred)buildTrue());
			}
		    }

		    // p or -p
		    for (int i = 0; !rewritten && i < args.length - 1; ++i) {
			Expr ci = ((Cvc3Pred)args[i]).getExpr();
			for (int j = 0; !rewritten && i < args.length; ++i) {
			    Expr cj = ((Cvc3Pred)args[i]).getExpr();
			    if (ci.equals(prover.notExpr(cj)) || cj.equals(prover.notExpr(ci))) {
				rewritten = true;
				node = transferLabels((Cvc3Pred)args[i], (Cvc3Pred)buildTrue());
				node = transferLabels((Cvc3Pred)args[j], node);
				//node = (Cvc3Pred)buildTrue();
			    }
			}
		    }
		}
		if (!rewritten) {
		    List children = new ArrayList();
		    List childrenExpr = new ArrayList();
                    // :TODO: consider omitted for label relevance
		    List omitted = new ArrayList();
		    for (int i = 0; i < args.length; ++i) {
			Expr ci = ((Cvc3Pred)args[i]).getExpr();
			if (!ci.equals(prover.falseExpr())
			    //||
			    //(rewriteCvc3 && prover.simplify(ci).equals(prover.trueExpr()))
			    ) {
			    children.add(args[i]);
			    childrenExpr.add(((Cvc3Pred)args[i]).getExpr());
			} else {
			    omitted.add(args[i]);
			}
		    }
		    if (children.size() == 0) {
			//:TODO: label
			//:TODO: cache, also for rewritting, simplify and of length 1
			node = (Cvc3Pred) buildFalse();
		    } else if (children.size() == 1) {
			node = (Cvc3Pred) children.get(0);
		    } else {
			node = new Cvc3Pred(prover.orExpr(childrenExpr));
			Iterator i = children.iterator();
			while (i.hasNext()) {
			    node.addChild((Cvc3Pred)i.next());
			}
			assert(node.getExpr().arity() == node.getChildren().size());
			assert(args.length >= node.getExpr().arity());
		    }
		    //node = transferLabels(omitted, node, false);
		}
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }


    public SPred buildNot(SPred _arg) {
	Cvc3Pred arg = (Cvc3Pred) _arg;
	Cvc3Prover.print("!buildNot: " + arg);
	try {
	    List key = Arrays.asList(new Object[] { "buildNot", arg });
	    if (escMap.containsKey(key)) {
		return (Cvc3Pred) escMap.get(key);
	    } else {
		Expr e1 = arg.getExpr();
		
		Cvc3Pred node = null;
		boolean rewritten = false;
		if (rewrite) {
		    //:TODO: move labels
		    if (e1.equals(prover.trueExpr())
			//			||
			//			(rewriteCvc3 && prover.simplify(e1).equals(prover.trueExpr()))
			) {
			assert(arg.isLabeled() == arg instanceof Cvc3Label);
			rewritten = true;
			node = transferLabels(arg, (Cvc3Pred) buildFalse(), true);
		    } else if (e1.equals(prover.falseExpr())
			       //			       ||
			       //			       (rewriteCvc3 && prover.simplify(e1).equals(prover.falseExpr()))
			       ) {
			assert(arg.isLabeled() == arg instanceof Cvc3Label);
			rewritten = true;
			node = transferLabels(arg, (Cvc3Pred) buildTrue(), true);
		    } else if (e1.isNot()) {
                        // e1 = not e2
			rewritten = true;
			// unwrap all labels
			Cvc3Pred unwrap = arg;
			while (unwrap instanceof Cvc3Label) {
			    assert (unwrap.getChildren().size() == 1);
			    unwrap = (Cvc3Pred) unwrap.getChildren().get(0);
			}
			// extract negated expression
			assert (unwrap.getExpr().equals(e1));
			assert (unwrap.getChildren().size() == 1);
                        // e2
			node = (Cvc3Pred) unwrap.getChildren().get(0);
                        // move labels from e1 to e2
			node = transferLabels(arg, node, true);
		    }
		}
		if (!rewritten) {
		    node = new Cvc3Pred(prover.notExpr(e1));
		    node.addChild(arg);
		}
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }

    public SPred buildTrue() {
	Cvc3Prover.print("!buildTrue: ");
	return escTrue;
    }
	
    public SPred buildFalse() {
	Cvc3Prover.print("!buildFalse: ");
	return escFalse;
    }

    public SPred buildDistinct(SAny[] terms) {
	Cvc3Prover.print("!buildDistinct: " + terms);
	for (int i = 0; i < terms.length; ++i) {
	    Cvc3Prover.print("  " + terms[i]);
	}

        /*
	List key = Arrays.asList(new Object[] { "buildDistinct", terms });
	if (escMap.containsKey(key)) {
	    return (Cvc3Pred) escMap.get(key);
	} else {
	    SPred[] pairs = new SPred[(terms.length * (terms.length - 1)) / 2];
	    int counter = 0;
	    for (int i = 0; i < terms.length - 1; ++i) {
		for (int j = i + 1; j < terms.length; ++j) {
		    pairs[counter] = buildAnyNE(terms[i], terms[j]);
		    ++counter;
		}
	    }
	    
	    Cvc3Pred node = (Cvc3Pred) buildAnd(pairs);
	    escMap.put(key, node);
            exprMap.put(node.getExpr(), node);
	    return node;
        }
        */

        try {
            // with an inductive datatype all java types are distinct
            if (optUseDatatype) {
                if (terms.length > 0
                    && ((Cvc3Any) terms[0]).getExpr().getType().equals(typeType)
                    )
                {
                    return buildTrue ();
                }
            }


            List children = new LinkedList();
            for (int i = 0; i < terms.length; ++i) {
                children.add(((Cvc3Any) terms[i]).getExpr());
            }
            Expr distinct = prover.distinctExpr(children);
            return (Cvc3Pred) wrapExpr(distinct);
        } catch (cvc3.Cvc3Exception e) {
            ErrorSet.fatal(e.toString());
            throw new Error(e);
        }
    }

    public SPred buildLabel(boolean pos, String name, SPred _pred) {
	Cvc3Pred pred = (Cvc3Pred) _pred;
	Cvc3Prover.print("!buildLabel: " + name + " " + pred);	
	try {
	    List key = Arrays.asList(new Object[] { "buildLabel", new Boolean(pos), name, pred });
	    if (escMap.containsKey(key)) {
		return (Cvc3Pred) escMap.get(key);
	    } else {
		Cvc3Pred node = null;
		// label not relevant: false LBLPOS or true LBLNEG
		if (
		    (pos && pred.getExpr().equals(prover.falseExpr()))
		    ||
		    (!pos && pred.getExpr().equals(prover.trueExpr()))
		    ) {
		    node = pred;
		}
		// create new label
		else {		
		    node = new Cvc3Label(pos, name, pred);
		}
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
        } catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }
	
    public SValue buildITE(SPred _cond, SValue _then_part, SValue _else_part) {
	Cvc3Pred cond = (Cvc3Pred) _cond;
	Cvc3Value then_part = (Cvc3Value) _then_part;
	Cvc3Value else_part = (Cvc3Value) _else_part;
	Cvc3Prover.print("!buildITE: " + cond + " " + then_part + " " + else_part);

	try {
	    List key = Arrays.asList(new Object[] { "buildITE", cond, then_part, else_part });
	    if (escMap.containsKey(key)) {
		return (Cvc3Value) escMap.get(key);
	    } else {
		// :TODO: labels not supported within ITE
		//assert(!cond.isLabeled());

		Cvc3Value node = null;
		if (rewrite && cond.getExpr().equals(prover.trueExpr())) {
                    node = then_part;
		} else if (rewrite && cond.getExpr().equals(prover.falseExpr())) {
                    node = then_part;
		} else {
		    Type t = unify(then_part.getExpr().getType(), else_part.getExpr().getType());
		    node = new Cvc3Value(prover.iteExpr(
						       cond.getExpr(),
						       cast(then_part.getExpr(), t),
						       cast(else_part.getExpr(), t)));
		}
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
	    
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }
    
    //@ requires \nonnullelements(vars);
    //@ requires pats != null ==> \nonnullelements(pats);
    //@ requires pats != null ==> \nonnullelements(nopats);
    public SPred buildForAll(QuantVar[] vars, SPred _body, STerm[][] pats, STerm[] nopats) {
	Cvc3Pred body = (Cvc3Pred) _body;
	Cvc3Prover.print("!buildForAll: " + vars + " " + body
	      + " " + pats + " " + nopats);
	try {
	    if (rewrite
		&&
		(body.getExpr().equals(prover.falseExpr())
		 || body.getExpr().equals(prover.trueExpr()))
		) {
		return _body;
	    }
	    List key = Arrays.asList(new Object[] { "buildForAll", vars, body, pats, nopats });
	    if (escMap.containsKey(key)) {
		return (Cvc3Pred) escMap.get(key);
	    } else {
		List varExprs = new ArrayList();
		for (int i = 0; i < vars.length; ++i) {
		    varExprs.add(getBoundVarExpr(vars[i]).getExpr());
		}

                /* change to cvc API
		// triggers: STerm[][]pats
		// - each list in pats represents a trigger
		// - each trigger consists of a set of expressions,
		//   i.e. a multi-trigger consists of more than one expression
		List triggers = new ArrayList();
		if (optManualTriggers && pats != null && pats.length > 0) {
		    Cvc3Prover.print("TRIGGERS: ");

		    //List triggers = new ArrayList();
		    for (int i = 0; i < pats.length; ++i) {
			assert(pats[i].length > 0);
			if (pats[i].length > 1) {
			    Cvc3Prover.print("MULTI-TRIGGER: ");
			    if (!optMultiTriggers) continue;
			}
			List trigger = new ArrayList();
			for (int j = 0; j < pats[i].length; ++j) {
			    Cvc3Prover.print("  TRIGGER: " + ((Cvc3Term)pats[i][j]).getExpr());
			    trigger.add(((Cvc3Term)pats[i][j]).getExpr());
			    //triggers.add(((Cvc3Term)pats[i][j]).getExpr());
			}
			triggers.add(trigger);
		    }
		}
		Expr e = prover.forallMultiExpr(varExprs, body.getExpr());
                */

		Expr e = prover.forallExpr(varExprs, body.getExpr());

                // add triggers
		if (optManualTriggers && pats != null && pats.length > 0) {
		    Cvc3Prover.print("TRIGGERS: ");

		    for (int i = 0; i < pats.length; ++i) {
			assert(pats[i].length > 0);
                        // multi-trigger
			if (pats[i].length > 1) {
			    Cvc3Prover.print("MULTI-TRIGGER: ");
			    if (!optMultiTriggers) continue;

                            List trigger = new ArrayList();
                            for (int j = 0; j < pats[i].length; ++j) {
                                Cvc3Prover.print("  TRIGGER: " + ((Cvc3Term)pats[i][j]).getExpr());
                                trigger.add(((Cvc3Term)pats[i][j]).getExpr());
                            }
                            prover.setMultiTrigger(e, trigger);
			}

                        // single trigger
                        else if (pats[i].length == 1) {
                            Cvc3Prover.print("  TRIGGER: " + ((Cvc3Term)pats[i][0]).getExpr());
                            prover.setTrigger(e, ((Cvc3Term)pats[i][0]).getExpr());
                        }
		    }
		}

		Cvc3Pred node = new Cvc3Pred(e);
		node.addChild(body);
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }

    //@ requires \nonnullelements(vars);
    public SPred buildExists(QuantVar[] vars, SPred _body) {
	Cvc3Pred body = (Cvc3Pred) _body;
	Cvc3Prover.print("!buildExists: " + vars + " " + body);
	try {
	    List key = Arrays.asList(new Object[] { "buildExists", vars, body });
	    if (escMap.containsKey(key)) {
		return (Cvc3Pred) escMap.get(key);
	    } else {
		List varExprs = new ArrayList();
		for (int i = 0; i < vars.length; ++i) {
		    varExprs.add(getBoundVarExpr(vars[i]).getExpr());
		}
		Cvc3Pred node = new Cvc3Pred(prover.existsExpr(varExprs, body.getExpr()));
		node.addChild(body);
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }
	

    // Arithmetic

    //@ requires predFirst <= realPredTag && realPredTag <= predLast;
    public SPred buildArithPred(int arithPredTag, SValue _arg1, SValue _arg2) {
	Cvc3Value arg1 = (Cvc3Value) _arg1;
	Cvc3Value arg2 = (Cvc3Value) _arg2;
	Cvc3Prover.print("!buildArithPred: " + arithPredTag + " " + arg1 + " " + arg2);
	try {
	    List key = Arrays.asList(new Object[] { "buildArithPred", new Integer(arithPredTag), arg1, arg2 });
	    if (escMap.containsKey(key)) {
		return (Cvc3Pred) escMap.get(key);
	    } else {
                Expr e1 = cast(arg1.getExpr(), typeReal); 
                Expr e2 = cast(arg2.getExpr(), typeReal);

		/* never happens
		while (sameCast(e1, e2) && isArithType(e1.getChild(0))) {
		    e1 = e1.getChild(0);
		    e2 = e2.getChild(0);
		    assert(isArithType(e1));
		    assert(isArithType(e2));
		}
		*/

		//Expr e1 = arg1.getExpr();
		//Expr e2 = arg2.getExpr();
		Cvc3Pred node = null;
		switch (arithPredTag) {
		case predGE: node = new Cvc3Pred(prover.geExpr(e1, e2)); break;
		case predLE: node = new Cvc3Pred(prover.leExpr(e1, e2)); break;
		case predGT: node = new Cvc3Pred(prover.gtExpr(e1, e2)); break;
		case predLT: node = new Cvc3Pred(prover.ltExpr(e1, e2)); break;
		case predEQ: node = (Cvc3Pred) buildAnyEQ(arg1, arg2); break;
		case predNE: node = (Cvc3Pred) buildAnyNE(arg1, arg2); break;
		default:
		    ErrorSet.fatal("buildRealPred"); assert(false);
		}

		return registerCvc3Pred(key, node);
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }


    //@ requires predFirst <= intPredTag && intPredTag <= predLast;
    public SPred buildIntPred(int intPredTag, SInt arg1, SInt arg2) {
	Cvc3Prover.print("!buildIntPred: " + intPredTag + " " + arg1 + " " + arg2);
	return buildArithPred(intPredTag, arg1, arg2);
    }
	
    //@ requires predFirst <= realPredTag && realPredTag <= predLast;
    public SPred buildRealPred(int realPredTag, SReal arg1, SReal arg2) {
	Cvc3Prover.print("!buildRealPred: " + realPredTag + " " + arg1 + " " + arg2);
	return buildArithPred(realPredTag, arg1, arg2);
    }

    //@ requires predFirst <= realPredTag && realPredTag <= predLast;
    public SBool buildArithBoolFun(int arithPredTag, SValue arg1, SValue arg2) {
	Cvc3Prover.print("!buildArithBoolFun: " + arithPredTag + " " + arg1 + " " + arg2);
	//assert(false);
	try {
	    List key = Arrays.asList(new Object[] { "buildArithBoolFun", new Integer(arithPredTag), arg1, arg2 });
	    if (escMap.containsKey(key)) {
		return (Cvc3Bool) escMap.get(key);
	    } else {
		Cvc3Pred p = (Cvc3Pred) buildArithPred(arithPredTag, arg1, arg2);
		Cvc3Bool node = (Cvc3Bool) wrapExpr(cast(p.getExpr(), sortBool));
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }

    //@ requires predFirst <= intPredTag && intPredTag <= predLast;
    public SBool buildIntBoolFun(int intPredTag, SInt arg1, SInt arg2) {
	Cvc3Prover.print("!buildIntBoolFun: " + intPredTag + " " + arg1 + " " + arg2);
	return buildArithBoolFun(intPredTag, arg1, arg2);
    }
  
    //@ requires predFirst <= realPredTag && realPredTag <= predLast;
    public SBool buildRealBoolFun(int realPredTag, SReal arg1, SReal arg2) {
	Cvc3Prover.print("!buildRealBoolFun: " + realPredTag + " " + arg1 + " " + arg2);
	return buildArithBoolFun(realPredTag, arg1, arg2);
    }
	
    //@ requires predFirst <= refPredTag && refPredTag <= predLast;
    public SBool buildRefBoolFun(int refPredTag, SRef arg1, SRef arg2) {
	Cvc3Prover.print("!buildRefBoolFun: " + refPredTag + " " + arg1 + " " + arg2);
	return new Cvc3Bool();
    }
    
    //@ requires funFirst <= intFunTag && intFunTag <= funLast;
    public Expr buildArithFun(int arithFunTag, SValue _arg1, SValue _arg2) {
	Cvc3Value arg1 = (Cvc3Value) _arg1;
	Cvc3Value arg2 = (Cvc3Value) _arg2;
	Cvc3Prover.print("!buildArithFun: " + arithFunTag + " " + arg1 + " " + arg2);
	try {
	    Expr e1 = cast(arg1.getExpr(), typeReal); 
	    Expr e2 = cast(arg2.getExpr(), typeReal);

	    /* never happens
	    while (sameCast(e1, e2) && isArithType(e1.getChild(0))) {
		System.out.println("SameCast: B");
		e1 = e1.getChild(0);
		e2 = e2.getChild(0);
		assert(isArithType(e1));
		assert(isArithType(e2));
	    }
	    */

	    switch (arithFunTag) {
	    case funADD: return prover.plusExpr(e1, e2);
	    case funSUB: return prover.minusExpr(e1, e2);
	    case funMUL: return prover.multExpr(e1, e2);
                /*            case funDIV:
                System.out.println("Unsupported: funDiv");
		throw new UnsupportedOperationException();
	    case funASR32:
                System.out.println("Unsupported: funASR32");
		throw new UnsupportedOperationException();
	    case funSL32:
                System.out.println("Unsupported: funSL32");
		throw new UnsupportedOperationException();
	    case funUSR32:
                System.out.println("Unsupported: funUSR32");
		throw new UnsupportedOperationException();
	    case funASR64:
                System.out.println("Unsupported: funASR64");
		throw new UnsupportedOperationException();
	    case funSL64:
                System.out.println("Unsupported: funSL64");
		throw new UnsupportedOperationException();
	    case funUSR64:
                System.out.println("Unsupported: funUSR64");
		throw new UnsupportedOperationException();
*/
                
            case funASR32: return prover.funExpr(mapFunSymbol(symASR32), e1, e2);
            case funSL32: return prover.funExpr(mapFunSymbol(symSL32), e1, e2);
            case funUSR32: return prover.funExpr(mapFunSymbol(symUSR32), e1, e2);
            case funASR64: return prover.funExpr(mapFunSymbol(symASR64), e1, e2);
            case funSL64: return prover.funExpr(mapFunSymbol(symSL64), e1, e2);
            case funUSR64: return prover.funExpr(mapFunSymbol(symUSR64), e1, e2);
                
	    default:
		//ErrorSet.fatal("buildArithFun"); assert(false);
		throw new Error("buildArithFun");
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }
    
    //@ requires funFirst <= intFunTag && intFunTag <= funLast;
    public SInt buildIntFun(int intFunTag, SInt arg1, SInt arg2) {
	Cvc3Prover.print("!buildIntFun: " + intFunTag + " " + arg1 + " " + arg2);
	try {
	    List key = Arrays.asList(new Object[] { "buildIntFun", new Integer(intFunTag), arg1, arg2 });
	    if (escMap.containsKey(key)) {
		return (Cvc3Int) escMap.get(key);
	    } else {
		Expr result;
		switch (intFunTag) {
		case funDIV: {
		    Expr exp1 = ((Cvc3Int) arg1).getExpr();
		    Expr exp2 = ((Cvc3Int) arg2).getExpr();
                    result = prover.funExpr(mapFunSymbol(symIntegralDiv), exp1, exp2);
		    break; }
		case funMOD: {
		    Expr exp1 = ((Cvc3Int) arg1).getExpr();
		    Expr exp2 = ((Cvc3Int) arg2).getExpr();
		    result = prover.funExpr(mapFunSymbol(symIntegralMod), exp1, exp2);
		    break; }
		default:
		    result = buildArithFun(intFunTag, arg1, arg2);
		}
		Cvc3Int node = new Cvc3Int(result);
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }

    //@ requires funFirst <= realFunTag && realFunTag <= funLast;
    public SReal buildRealFun(int realFunTag, SReal arg1, SReal arg2) {
	Cvc3Prover.print("!buildRealFun: " + realFunTag + " " + arg1 + " " + arg2);
	try {
	    List key = Arrays.asList(new Object[] { "buildRealFun", new Integer(realFunTag), arg1, arg2 });
	    if (escMap.containsKey(key)) {
		return (Cvc3Real) escMap.get(key);
	    } else {
		Expr result;
		switch (realFunTag) {
		case funDIV: {
		    Expr exp1 = ((Cvc3Real) arg1).getExpr();
		    Expr exp2 = ((Cvc3Real) arg2).getExpr();
		    result = prover.funExpr(mapFunSymbol(symFloatingDiv), exp1, exp2);
		    // :TODO: result = prover.divideExpr(exp1, exp2);
		    break; }
		case funMOD: {
		    Expr exp1 = ((Cvc3Real) arg1).getExpr();
		    Expr exp2 = ((Cvc3Real) arg2).getExpr();
		    result = prover.funExpr(mapFunSymbol(symFloatingMod), exp1, exp2);
		    break; }
		default:
		    result = buildArithFun(realFunTag, arg1, arg2);
		}
		Cvc3Real node = new Cvc3Real(result);
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }
    
    public Expr buildArithFun(int arithFunTag, SValue _arg1) {
	Cvc3Value arg1 = (Cvc3Value) _arg1;
	Cvc3Prover.print("!buildArithFun: " + arithFunTag + " " + arg1);
	try {
	    Expr e = cast(arg1.getExpr(), typeReal);
	    switch (arithFunTag) {
	    case funNEG: return prover.uminusExpr(e);
	    default:
		ErrorSet.fatal("buildArithFun"); assert(false);
		throw new Error("buildArithFun");
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }

    //@ requires funUniFirst <= intFunTag && intFunTag <= funUniLast;
    public SInt buildIntFun(int intFunTag, SInt arg1) {
	Cvc3Prover.print("!buildIntFun: " + intFunTag + " " + arg1);
        List key = Arrays.asList(new Object[] { "buildIntFun", new Integer(intFunTag), arg1 });
        if (escMap.containsKey(key)) {
            return (Cvc3Int) escMap.get(key);
        } else {
            Cvc3Int node = new Cvc3Int(buildArithFun(intFunTag, arg1));
            escMap.put(key, node);
            exprMap.put(node.getExpr(), node);
            return node;
        }
    }
	
    //@ requires funUniFirst <= realFunTag && realFunTag <= funUniLast;
    public SReal buildRealFun(int realFunTag, SReal arg1) {
	Cvc3Prover.print("!buildRealFun: " + realFunTag + " " + arg1);
        List key = Arrays.asList(new Object[] { "buildRealFun", new Integer(realFunTag), arg1 });
        if (escMap.containsKey(key)) {
            return (Cvc3Real) escMap.get(key);
        } else {
            Cvc3Real node = new Cvc3Real(buildArithFun(realFunTag, arg1));
            escMap.put(key, node);
            exprMap.put(node.getExpr(), node);
            return node;
        }
    }
	

    // Literals

    public SBool buildBool(boolean b) {
	Cvc3Prover.print("!buildBool: " + b);
	if (b) {
	    return escBTrue;
	} else {
	    return escBFalse;
	}
    }

    public SInt buildInt(long n) {
	Cvc3Prover.print("!buildInt: " + n);
	try {
	    List key = Arrays.asList(new Object[] { "buildInt", new Long(n) });
	    if (escMap.containsKey(key)) {
		return (Cvc3Int) escMap.get(key);
	    } else {
		Cvc3Int i = new Cvc3Int(prover.ratExpr(Long.toString(n)));
		escMap.put(key, i);
                exprMap.put(i.getExpr(), i);
		return i;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }

    public SReal buildReal(double f) {
	Cvc3Prover.print("!buildReal: " + f);
	try {
	    List key = Arrays.asList(new Object[] { "buildReal", new Double(f) });
	    if (escMap.containsKey(key)) {
		return (Cvc3Real) escMap.get(key);
	    } else {
		Cvc3Real r = new Cvc3Real(cast(prover.ratExpr(Double.toString(f)), typeReal));
		assert(!escMap.containsKey(key));
		escMap.put(key, r);
                exprMap.put(r.getExpr(), r);
		return r;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }

    public SRef buildNull() {
	Cvc3Prover.print("!buildNull: ");
	return escNull;
    }


    // Maps
	
    public SValue buildSelect(SMap _map, SValue _idx) {
	Cvc3Map map = (Cvc3Map) _map;
	Cvc3Value idx = (Cvc3Value) _idx;
	Cvc3Prover.print("!buildSelect: " + map + " " + idx);
	try {
	    List key = Arrays.asList(new Object[] { "buildSelect", map, idx });
	    if (escMap.containsKey(key)) {
		return (Cvc3Value) escMap.get(key);
	    } else {
		assert(map.getExpr().getType().arity() == 2);
		Type tIndex = map.getExpr().getType().getChild(0);
		Expr expr = prover.readExpr(map.getExpr(), cast(idx.getExpr(), tIndex));
		
		Cvc3Value node = (Cvc3Value) wrapExpr(expr);
		//Cvc3Value node = (Cvc3Value) wrapAnyExpr(expr, map.sort???);
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }

    public SMap buildStore(SMap _map, SValue _idx, SValue _val) {
	Cvc3Map map = (Cvc3Map) _map;
	Cvc3Value idx = (Cvc3Value) _idx;
	Cvc3Value val = (Cvc3Value) _val;
	Cvc3Prover.print("!buildStore: " + map + " " + idx + " " + val);
	try {
	    List key = Arrays.asList(new Object[] { "buildStore", map, idx, val });
	    if (escMap.containsKey(key)) {
		return (Cvc3Map) escMap.get(key);
	    } else {
		Type type = map.getExpr().getType();
		assert(type.arity() == 2);
		Type tIndex = type.getChild(0);
		Type tValue = type.getChild(1);
		Cvc3Map node = new Cvc3Map(
		  prover.writeExpr(
				   map.getExpr(),
				   cast(idx.getExpr(), tIndex),
				   cast(val.getExpr(), tValue)));
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }
	
	
    // Equality

    public SPred buildAnyEQ(SAny _arg1, SAny _arg2) {
	Cvc3Any arg1 = (Cvc3Any) _arg1;
	Cvc3Any arg2 = (Cvc3Any) _arg2;
	Cvc3Prover.print("!buildAnyEQ: " + arg1 + " " + arg2);

	try {
	    List key = Arrays.asList(new Object[] { "buildAnyEQ", _arg1, _arg2 });
	    if (escMap.containsKey(key)) {
		return (Cvc3Pred) escMap.get(key);
	    } else {
		Expr e1 = arg1.getExpr();
		Expr e2 = arg2.getExpr();		
		Type t = unify(e1.getType(), e2.getType());
		
		Expr e1c = cast(e1, t);
		Expr e2c = cast(e2, t);
		while (sameCast(e1c, e2c)) {
		    e1c = e1c.getChild(0);
		    e2c = e2c.getChild(0);
		}

		Cvc3Pred node = null;
		boolean rewritten = false;
		if (true || rewrite) {
		    if (e1c.equals(e2c)) {
			//:TODO: move labels
			rewritten = true;
			node = (Cvc3Pred) buildTrue();
		    } else if (rewriteCvc3 && prover.simplify(prover.eqExpr(e1c, e2c)).equals(prover.trueExpr())) {
			rewritten = true;
			assert(rewritten || !prover.simplify(prover.eqExpr(e1c, e2c)).equals(prover.trueExpr()));
			node = (Cvc3Pred) buildTrue();
		    } else if (rewriteCvc3 && prover.simplify(prover.eqExpr(e1c, e2c)).equals(prover.falseExpr())) {
			rewritten = true;
			assert(rewritten || !prover.simplify(prover.eqExpr(e1c, e2c)).equals(prover.falseExpr()));
			node = (Cvc3Pred) buildFalse();
		    }
		}
		if (!rewritten) {
		    node = new Cvc3Pred(prover.eqExpr(e1c, e2c));
		}

		return registerCvc3Pred(key, node);
		
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }

    public SPred buildAnyNE(SAny arg1, SAny arg2) {
	Cvc3Prover.print("!buildAnyNE: " + arg1 + " " + arg2);

	return buildNot(buildAnyEQ(arg1, arg2));
    }
	
	
    // Type conversions
    public SValue buildValueConversion(Sort from, Sort to, SValue _value) {
	Cvc3Value value = (Cvc3Value) _value;
	Cvc3Prover.print("!buildValueConversion: " + from + " " + to + " " + value);
	try {
	    List key = Arrays.asList(new Object[] { "buildValueConversion", from, to, value });
	    if (escMap.containsKey(key)) {
		return (Cvc3Value) escMap.get(key);
	    } else {
		Cvc3Value node = (Cvc3Value) wrapExpr(cast(value.getExpr(), to), to);
		escMap.put(key, node);
                exprMap.put(node.getExpr(), node);
		return node;
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }

    // Type conversions
    public SValue buildPredToBool(SPred _pred) {
	Cvc3Prover.print("!buildPredToBool: " + _pred);
        return buildITE(_pred, buildBool(true), buildBool(false));
    }

    public SPred buildIsTrue(SBool _val) {
	Cvc3Bool val = (Cvc3Bool) _val;
	Cvc3Prover.print("!buildIsTrue: " + val);
	try {
	    List key = Arrays.asList(new Object[] { "buildIsTrue", val });
	    if (escMap.containsKey(key)) {
		return (Cvc3Pred) escMap.get(key);
	    } else {
		Expr e = prover.eqExpr(((Cvc3Bool)buildBool(true)).getExpr(), ((Cvc3Bool)val).getExpr());
		return registerCvc3Pred(key, new Cvc3Pred(e));
	    }
	} catch (cvc3.Cvc3Exception e) {
	    ErrorSet.fatal(e.toString());
	    throw new Error(e);
	}
    }


    // Mobius stuff
    public SPred buildNewObject(SMap oldh, SAny type, SMap heap, SValue r) {
	throw new UnsupportedOperationException();
    }
	
    public SAny buildSort(Sort s) {
	throw new UnsupportedOperationException();
    }
	
    public SValue buildDynSelect(SMap map, SValue obj, SAny field) {
	throw new UnsupportedOperationException();
    }

    public SRef buildDynLoc(SMap map, SValue obj, SAny field) {
	throw new UnsupportedOperationException();
    }
	
    public SMap buildDynStore(SMap map, SValue obj, SAny field, SValue val) {
	throw new UnsupportedOperationException();
    }
	
    public SValue buildArrSelect(SMap map, SRef obj, SInt idx) {
	throw new UnsupportedOperationException();
    }
	
    public SMap buildArrStore(SMap map, SRef obj, SInt idx, SValue val) {
	throw new UnsupportedOperationException();
    }
	
    public SPred buildNewArray(SMap oldh, SAny type, SMap heap, SRef r, SInt len) {
	throw new UnsupportedOperationException();
    }
	
    public SPred buildAssignCompat(SMap map, SValue val, SAny type) {
	throw new UnsupportedOperationException();
    }

    public SPred buildInv(SMap map, SValue val, SAny type) {
	throw new UnsupportedOperationException();
    }

    public SPred buildIsAlive(SMap map, SRef ref) {
	throw new UnsupportedOperationException();
    }

    public SPred buildAssignPred(SMap map, SMap map_pre, SRef target, SRef loc) {
	throw new UnsupportedOperationException();
    }



    //
    // labels
    //

    // from        : if 'from' is a label then move it to wrap 'to'
    // to          : target expression
    // pos         : we are interested in labels of this polarity only
    // flipPolarity: flip the polarity of a label, i.e. when it is moved over a not simplification
    public Cvc3Pred transferLabels(Cvc3Pred from, Cvc3Pred to, boolean flipPolarity) throws Cvc3Exception  {
        // move from to to if from is a label
	if (from instanceof Cvc3Label) {
	    //assert(false);
	    Cvc3Label label = (Cvc3Label) from;
	    //Cvc3Pred transferred = transferLabels(label.getPred(), to, flipPolarity);
	    boolean polarity = label.isPos();
            if (flipPolarity) polarity = !polarity;
            to = (Cvc3Pred) buildLabel(polarity, label.getName(), to);
	}

        // move labels in subexpressions of from to to
        if (from.isLabeled()) {
            if (from.getExpr().isNot()) {
                flipPolarity = !flipPolarity;
            }

            List children = from.getChildren();
            Iterator i = children.iterator();
            while (i.hasNext()) {
                Cvc3Term child = (Cvc3Term) i.next();
                if (child instanceof Cvc3Pred) {
                    transferLabels((Cvc3Pred) child, to, flipPolarity);
                }
            }
        }

        return to;
    }

    public Cvc3Pred transferLabels(Cvc3Pred from, Cvc3Pred to) throws Cvc3Exception {
        return transferLabels(from, to, false);
    }



    // when using inductive datatypes,
    // guard all occurrences of valueToX(elems[a][i])
    // by is_javaX AND
    class Pair {
        Pair(Object fst, Object snd) {
            this.fst = fst;
            this.snd = snd;
        }
        public Object fst;
        public Object snd;
    }



    void getUnboundQuantifiedVars(List vars, Expr expr) throws Cvc3Exception {
        if (expr.isBoundVar()) {
            vars.add(expr);
        }

        if (expr.isClosure()) {
            getUnboundQuantifiedVars(vars, expr.getBody());
            List bound = expr.getVars();
            Iterator i = bound.iterator();
            while (i.hasNext()) {
                vars.remove(i.next());
            }
        } else {
            for (int i = 0; i < expr.arity(); ++i) {
                getUnboundQuantifiedVars(vars, expr.getChild(i));
            }
        }
        
    }

    protected Expr guardTraverse(Expr expr_, HashMap cache, List guards) throws Cvc3Exception {
        Expr expr = expr_;
        if (cache.containsKey(expr)) {
            return (Expr) cache.get(expr);
        }

        if (expr.isTerm()) {
            cache.put(expr, expr);
            return expr;
        }

        // traverse subexpressions
        if (expr.isClosure()) {
            Expr body = guardTraverse(expr.getBody(), cache, guards);
            if (!body.equals(expr.getBody())) {
                if (expr.isExists()) {
                    expr = prover.existsExpr(expr.getVars(), body);
                } else if (expr.isForall()) {
                    expr = prover.forallExpr(expr.getVars(), body);
                } else if (expr.isLambda()) {
                    expr = prover.funExpr(expr.getOp(), body);
                } else {
                    assert (false);
                }

                // must use EscJava term structure to get labels right
                Cvc3Pred pred = new Cvc3Pred(expr);
                Cvc3Term child = (Cvc3Term) exprMap.get(body);
                assert (child != null);
                if (child instanceof Cvc3Pred) {
                    pred.addChild((Cvc3Pred) child);
                }
                exprMap.put(pred.getExpr(), pred);
            }
            
        } else {
            //if (expr.isBoolConnective() || expr.isApply()) {
            List children = new LinkedList();
            boolean changed = false;
            for (int i = 0; i < expr.arity(); ++i) {
                Expr guarded = guardTraverse(expr.getChild(i), cache, guards);
                children.add(guarded);
                changed = changed || !guarded.equals(expr.getChild(i));
            }
            if (changed) {
                //System.out.println("FUN:");
                //System.out.println("EXPR: " + expr.toString());

                expr = prover.funExpr(expr.getOp(), children);

                // must use EscJava term structure to get labels right
                Cvc3Pred pred = new Cvc3Pred(expr);
                Iterator i = children.iterator();
                while (i.hasNext()) {
                    Cvc3Term child = (Cvc3Term) exprMap.get(i.next());
                    assert (child != null);
                    if (child instanceof Cvc3Pred) {
                        pred.addChild((Cvc3Pred) child);
                    }
                }
                exprMap.put(pred.getExpr(), pred);
            }
        }

        // check if expression needs to be guarded
	if (expr.isApply()) {
            Op op = expr.getOp();
            if (op.equals(valueToBool) || op.equals(valueToInt)
                || op.equals(valueToReal) || op.equals(valueToRef)) {

                // ? elems[a][i]
                assert (expr.arity() == 1);
                Expr child = expr.getChild(0);
                //System.out.println(child);
                if (child.isRead()) {
                    assert (expr.arity() == 1);
                    child = child.getChild(0);
                    //System.out.println(child);
                    if (child.isRead()) {
                        assert (expr.arity() == 1);
                        if (child.getChild(0).getType().equals(typeElems)) {
                            List unboundQuantifiedVars = new LinkedList();
                            getUnboundQuantifiedVars(unboundQuantifiedVars, expr);

                            String tester = null;
                            if (op.equals(valueToBool)) {
                                tester = "javaBool";
                            } else if (op.equals(valueToInt)) {
                                tester = "javaInt";
                            } else if (op.equals(valueToReal)) {
                                tester = "javaReal";
                            } else if (op.equals(valueToRef)) {
                                tester = "javaRef";
                            }

                            Expr guard = prover.datatypeTestExpr(tester, expr.getChild(0));

                            guards.add(new Pair(unboundQuantifiedVars, guard));
                        }
                    }
                }
            }
        }

        // add guard
        if (expr.isQuantifier()) {
            List boundVars = expr.getVars();
            Iterator i = boundVars.iterator();
            while (i.hasNext()) {
                Expr v = (Expr) i.next();

                Iterator j = guards.iterator();
                while (j.hasNext()) {
                    Pair guard_pair = (Pair) j.next();
                    List vars = (List) guard_pair.fst;
                    if (vars.contains(v)) {
                        guards.remove(guard_pair);

                        // must use EscJava term structure to get labels right
                        Cvc3Pred guard = (Cvc3Pred) exprMap.get((Expr)guard_pair.snd);
                        assert (guard != null);
                        Cvc3Pred body = (Cvc3Pred) exprMap.get(expr.getBody());
                        assert (body != null);
                        Cvc3Pred guarded = (Cvc3Pred) buildImplies(guard, body);

                        if (expr.isExists()) {
                            expr = prover.existsExpr(boundVars, guarded.getExpr());
                        } else if (expr.isForall()) {
                            expr = prover.forallExpr(boundVars, guarded.getExpr());
                        } else {
                            assert (false);
                        }

                        Cvc3Pred pred = new Cvc3Pred(expr);
                        pred.addChild(guarded);
                        exprMap.put(pred.getExpr(), pred);
                    }
                }
            }
        }

        cache.put(expr, expr);
        return expr;
    }


    // when using inductive datatypes,
    // guard all occurrences of valueToX(elems[a][i])
    // by is_javaX AND
    public Cvc3Pred guard(Cvc3Pred pred) throws Cvc3Exception {
        if (!optUseDatatype) {
            return pred;
        }

        Expr expr = pred.getExpr();
        HashMap cache = new HashMap();
        List guards = new LinkedList();
        Cvc3Pred guarded = (Cvc3Pred) exprMap.get(guardTraverse(expr, cache, guards));
        assert (guarded != null);

        Iterator i = guards.iterator();
        while (false && i.hasNext()) {
            Pair guard_pair = (Pair) i.next();
            assert (((List)guard_pair.fst).isEmpty());
            Cvc3Pred guard = (Cvc3Pred) exprMap.get((Expr)guard_pair.snd);
            assert (guard != null);
            guarded = (Cvc3Pred) buildImplies(guard, guarded);
        }

        return guarded;
    }

}
