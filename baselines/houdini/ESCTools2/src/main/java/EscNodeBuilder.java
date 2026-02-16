package escjava.sortedProver;

import javafe.util.ErrorSet;
import escjava.ast.TagConstants;
import escjava.sortedProver.NodeBuilder.FnSymbol;
import escjava.sortedProver.NodeBuilder.PredSymbol;
import escjava.sortedProver.NodeBuilder.Sort;

/*@ non_null_by_default @*/
public abstract class EscNodeBuilder extends NodeBuilder
{
	final public Sort sortShape = registerSort("shape", sortAny);
	final public Sort sortTime = registerSort("time", sortInt);
	// it seems instances of java.lang.Class are treated as equal to
	// values of sort type...
	final public Sort sortType = registerSort("type", sortRef);//= sortRef; // registerSort("type", sortAny);
	final public Sort sortLock = sortRef; //registerSort("lock", sortRef);
  final public Sort sortField = registerSort("field", sortMap);
  
	final public Sort sortIntField = registerMapSort(sortRef, sortInt, sortField);
	final public Sort sortRealField = registerMapSort(sortRef, sortReal, sortField);
	final public Sort sortBoolField = registerMapSort(sortRef, sortBool, sortField);
	final public Sort sortRefField = registerMapSort(sortRef, sortRef, sortField);

  
	final public Sort sortOwner = registerMapSort(sortRef, sortRef, sortField); // RefField
	
	final public Sort sortArray = sortRef;
	final public Sort sortArrayValue = registerMapSort(sortInt, sortValue);
	final public Sort sortElems = registerMapSort(sortRef, sortArrayValue);
	
	final public Sort sortLockSet = registerMapSort(sortLock, sortBool); // BoolField
		
	// TODO reflect distinction between Array and ArrayValue in types of functions
	
	// Don't make these final, so deriving class can redefine them
	// if it wishes so.
	
	// Equality
	public PredSymbol symRefEQ = registerPredSymbol("refEQ", new Sort[] { sortRef, sortRef }, TagConstants.REFEQ);
	public PredSymbol symRefNE = registerPredSymbol("refNEQ", new Sort[] { sortRef, sortRef }, TagConstants.REFNE);
	// Built-in orders
	public PredSymbol symAllocLT = registerPredSymbol("<", new Sort[] { sortTime, sortTime }, TagConstants.ALLOCLT);
	public PredSymbol symAllocLE = registerPredSymbol("<=", new Sort[] { sortTime, sortTime }, TagConstants.ALLOCLE);
	public PredSymbol symLockLT = registerPredSymbol("lockLT", new Sort[] { sortLock, sortLock }, TagConstants.LOCKLT);
	public PredSymbol symLockLE = registerPredSymbol("lockLE", new Sort[] { sortLock, sortLock }, TagConstants.LOCKLE);
	public PredSymbol symTypeEQ = registerPredSymbol("typeEQ", new Sort[] { sortType, sortType }, TagConstants.TYPEEQ);
	public PredSymbol symTypeNE = registerPredSymbol("typeNEQ", new Sort[] { sortType, sortType }, TagConstants.TYPENE);
	public PredSymbol symTypeLE = registerPredSymbol("<:", new Sort[] { sortType, sortType }, TagConstants.TYPELE);
	// Dynamic types
	public PredSymbol symIs = registerPredSymbol("is", new Sort[] { sortValue, sortType }, TagConstants.IS);
	public FnSymbol symCast = registerFnSymbol("cast", new Sort[] { sortValue, sortType }, sortValue, TagConstants.CAST);
	public FnSymbol symTypeOf = registerFnSymbol("typeof", new Sort[] { sortRef }, sortType, TagConstants.TYPEOF);	
	// Allocation status
	public PredSymbol symIsAllocated = registerPredSymbol("isAllocated", new Sort[] { sortRef, sortTime }, TagConstants.ISALLOCATED);
	public FnSymbol symEClosedTime = registerFnSymbol("eClosedTime", new Sort[] { sortElems }, sortTime, TagConstants.ECLOSEDTIME); 
	public FnSymbol symFClosedTime = registerFnSymbol("fClosedTime", new Sort[] { sortField }, sortTime, TagConstants.FCLOSEDTIME);
	// The as-trick
	public FnSymbol symAsChild = registerFnSymbol("asChild", new Sort[] { sortType, sortType }, sortType );
	public FnSymbol symClassDown = registerFnSymbol("classDown", new Sort[] { sortType, sortType }, sortType );
	public FnSymbol symAsElems = registerFnSymbol("asElems", new Sort[] { sortElems }, sortElems, TagConstants.ASELEMS);
	public FnSymbol symAsField = registerFnSymbol("asField", new Sort[] { sortField, sortType }, sortField, TagConstants.ASFIELD);
	public FnSymbol symAsLockSet = registerFnSymbol("asLockSet", new Sort[] { sortLockSet }, sortLockSet, TagConstants.ASLOCKSET);
	// Arrays
	public FnSymbol symArrayLength = registerFnSymbol("arrayLength", new Sort[] { sortArray }, sortInt, TagConstants.ARRAYLENGTH);
	public FnSymbol symArrayShapeOne = registerFnSymbol("arrayShapeOne", new Sort[] { sortInt }, sortShape, TagConstants.ARRAYSHAPEONE);
	public FnSymbol symArrayShapeMore = registerFnSymbol("arrayShapeMore", new Sort[] { sortInt, sortShape }, sortShape, TagConstants.ARRAYSHAPEMORE);
	public FnSymbol symArrayParent = registerFnSymbol("arrayParent", new Sort[] { sortArray }, sortArray);
	public FnSymbol symArrayPosition = registerFnSymbol("arrayPosition", new Sort[] { sortArray }, sortInt);
	public PredSymbol symArrayFresh = registerPredSymbol("arrayFresh", 
			new Sort[] { sortArray, sortTime, sortTime, sortElems, sortShape, sortType, sortValue }, TagConstants.ARRAYFRESH);
	public PredSymbol symIsNewArray = registerPredSymbol("isNewArray", new Sort[] { sortArray }, TagConstants.ISNEWARRAY);
	public FnSymbol symUnset = registerFnSymbol("unset", new Sort[] { sortArrayValue, sortInt, sortInt }, sortArrayValue, TagConstants.UNSET);
	
	// stuff unhandled by the T* interface
	public FnSymbol symArray = registerFnSymbol("_array", new Sort[] { sortType }, sortType);	
    public FnSymbol symIntern = registerFnSymbol("|intern:|", new Sort[] { sortInt, sortInt }, sortString, TagConstants.INTERN);
    //:ALEX: was PredSymbol:
    // - is used as predicate and function
    // - Lifter.saveBgSymbol assumes that it is a function
    public FnSymbol symInterned = registerFnSymbol("|interned:|", new Sort[] { sortString }, sortBool, TagConstants.INTERNED);
    public FnSymbol symStringCat = registerFnSymbol("stringCat", new Sort[] { sortString, sortString, sortTime }, sortString, TagConstants.STRINGCAT);
    // the 3rd argument seems to be sometimes int and sometimes string
    public PredSymbol symStringCatP = registerPredSymbol("stringCatP", new Sort[] { sortString, sortString, sortValue, sortTime, sortTime }, TagConstants.STRINGCATP);
    public PredSymbol symNonNullElems = registerPredSymbol("nonnullelements", new Sort[] { sortArray, sortElems }, TagConstants.ELEMSNONNULL);
    public FnSymbol symElemType = registerFnSymbol("elemtype", new Sort[] { sortType }, sortType, TagConstants.ELEMTYPE);
    public FnSymbol symMax = registerFnSymbol("max", new Sort[] { sortLockSet }, sortLock, TagConstants.MAX);
    public FnSymbol symArrayMake = registerFnSymbol("arrayMake", new Sort[] { sortTime, sortTime, sortShape, sortType, sortValue }, sortArray, TagConstants.ARRAYMAKE);
    public FnSymbol symClassLiteral = registerFnSymbol("classLiteral", new Sort[] { sortType }, sortRef, TagConstants.CLASSLITERALFUNC);


    public /*@ nullable @*/FnSymbol mapFnSymbolTo(EscNodeBuilder other, FnSymbol sym)
    {
    	if (sym == symRefEQ) return other.symRefEQ;
    	else if (sym == symRefNE) return other.symRefNE;
    	else if (sym == symAllocLT) return other.symAllocLT;
    	else if (sym == symAllocLE) return other.symAllocLE;
    	else if (sym == symLockLT) return other.symLockLT;
    	else if (sym == symLockLE) return other.symLockLE;
    	else if (sym == symTypeEQ) return other.symTypeEQ;
    	else if (sym == symTypeNE) return other.symTypeNE;
    	else if (sym == symTypeLE) return other.symTypeLE;
    	else if (sym == symIs) return other.symIs;
    	else if (sym == symCast) return other.symCast;
    	else if (sym == symTypeOf) return other.symTypeOf;
    	else if (sym == symIsAllocated) return other.symIsAllocated;
    	else if (sym == symEClosedTime) return other.symEClosedTime;
    	else if (sym == symFClosedTime) return other.symFClosedTime;
    	else if (sym == symAsElems) return other.symAsElems;
    	else if (sym == symAsField) return other.symAsField;
    	else if (sym == symAsLockSet) return other.symAsLockSet;
    	else if (sym == symArrayLength) return other.symArrayLength;
    	else if (sym == symArrayShapeOne) return other.symArrayShapeOne;
    	else if (sym == symArrayShapeMore) return other.symArrayShapeMore;
    	else if (sym == symArrayFresh) return other.symArrayFresh;
    	else if (sym == symIsNewArray) return other.symIsNewArray;
    	else if (sym == symUnset) return other.symUnset;
    	else if (sym == symArray) return other.symArray;
    	else if (sym == symIntern) return other.symIntern;
    	else if (sym == symInterned) return other.symInterned;
    	else if (sym == symStringCat) return other.symStringCat;
    	else if (sym == symStringCatP) return other.symStringCatP;
    	else if (sym == symNonNullElems) return other.symNonNullElems;
    	else if (sym == symElemType) return other.symElemType;
    	else if (sym == symMax) return other.symMax;
    	else if (sym == symArrayMake) return other.symArrayMake;
    	else if (sym == symClassLiteral) return other.symClassLiteral;
    	//ErrorSet.fatal("cannot map " + sym + " from " + this + " to " + other);
    	return null;
    }
    
    public Sort canonicalizeSort(Sort s)
    {
    	s = s.theRealThing();
    	Sort from = s.getMapFrom();
    	if (from != null) from = from.theRealThing();
    	if (from == null) {
    		return s;
    	} else if (from == sortRef) {
    		Sort to = s.getMapTo().theRealThing();
    		if (to == sortRef) return sortRefField;
    		if (to == sortInt) return sortIntField;
    		if (to == sortBool) return sortBoolField;
    		if (to == sortReal) return sortRealField;
    		to = canonicalizeSort(to);
    		if (to == sortArrayValue) return sortElems;
    		return s;
    	} else if (from == sortInt) {
    		Sort to = s.getMapTo().theRealThing();
    		if (to == sortValue) return sortArrayValue;
    		return s;
    	} else {
    		return s;
    	}
    }

	public Sort mapSortTo(EscNodeBuilder other, Sort sort)
	{
		sort = canonicalizeSort(sort);
		if (sort == sortIntField) return other.sortIntField;
		else if (sort == sortRealField) return other.sortRealField;
		else if (sort == sortBoolField) return other.sortBoolField;
		else if (sort == sortRefField) return other.sortRefField;
		else if (sort == sortArrayValue) return other.sortArrayValue;
		else if (sort == sortElems) return other.sortElems;
		else if (sort == sortShape) return other.sortShape;
		else if (sort == sortTime) return other.sortTime;
		else if (sort == sortPred) return other.sortPred;
		else if (sort == sortAny) return other.sortAny;
		else if (sort == sortValue) return other.sortValue;
		else if (sort == sortBool) return other.sortBool;
		else if (sort == sortInt) return other.sortInt;
		else if (sort == sortReal) return other.sortReal;
		else if (sort == sortRef) return other.sortRef;
		else if (sort == sortMap) return other.sortMap;
    else if (sort == sortType) return other.sortType;
    else if (sort == sortField) return other.sortField;
		ErrorSet.fatal("cannot map " + sort + " from " + this + " to " + other);
		return null;
	}

}
