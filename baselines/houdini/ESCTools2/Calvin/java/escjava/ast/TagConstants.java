/* Copyright Hewlett-Packard, 2002 */

package escjava.ast;

import javafe.ast.Identifier;
import javafe.util.Assert;

import escjava.prover.Atom;

public class TagConstants
  extends javafe.tc.TagConstants
  implements GeneratedTags
{
  //// Tags for new binary operators
  public static final int IMPLIES = escjava.ast.GeneratedTags.LAST_TAG + 1;
  public static final int SUBTYPE = IMPLIES + 1;  
  public static final int CHOICE = SUBTYPE + 1;

  //// Tag for "->" operator for map types
  public static final int MAPSTO = CHOICE + 1;

  //// Tags for new literal expressions
  public static final int SYMBOLLIT = MAPSTO + 1;

  //// Tags for new primitive types
  public static final int ANY = SYMBOLLIT + 1;
  public static final int TYPECODE = ANY + 1;
  public static final int LOCKSET = TYPECODE + 1;
    
  //// Tags for guarded commands
  public static final int ASSERTCMD = LOCKSET + 1;
  public static final int ASSUMECMD = ASSERTCMD + 1;
  public static final int CHOOSECMD = ASSUMECMD + 1;
  public static final int RAISECMD = CHOOSECMD + 1;
  public static final int SKIPCMD = RAISECMD + 1;
  public static final int TRYCMD = SKIPCMD + 1;

  //// Tags for ESCJ keywords
  public static final int FIRSTESCKEYWORDTAG = TRYCMD + 1;
  public static final int ACTION = FIRSTESCKEYWORDTAG;
  public static final int ALSO_ENSURES = ACTION + 1;
  public static final int ALSO_EXSURES = ALSO_ENSURES + 1;
  public static final int ALSO_MODIFIES = ALSO_EXSURES + 1;
  public static final int ALSO_REQUIRES = ALSO_MODIFIES + 1;
  public static final int ASSUME = ALSO_REQUIRES + 1;
  public static final int ASSERT = ASSUME + 1;
  public static final int AXIOM = ASSERT + 1;
  public static final int DEFINED_IF = AXIOM + 1;
  public static final int DTTFSA = DEFINED_IF + 1;
  public static final int ENSURES = DTTFSA + 1;
  public static final int ELEMS_UNWRITABLE_BY_ENV_IF = ENSURES + 1;
  public static final int ELEMSNONNULL = ELEMS_UNWRITABLE_BY_ENV_IF + 1; // Function
  public static final int ELEMTYPE = ELEMSNONNULL + 1; // Function
  public static final int EXISTS = ELEMTYPE + 1;
  public static final int EXSURES = EXISTS + 1;
  public static final int FRESH = EXSURES + 1; // Non-GCE function
  public static final int FORALL = FRESH + 1;
  public static final int GHOST = FORALL + 1;
  public static final int GLOBAL_INVARIANT = GHOST + 1;
  public static final int UNWRITABLE_BY_ENV_IF = GLOBAL_INVARIANT + 1;
  public static final int HELPER = UNWRITABLE_BY_ENV_IF + 1;
  public static final int INVARIANT = HELPER + 1;
  public static final int LBLPOS = INVARIANT + 1;
  public static final int LBLNEG = LBLPOS + 1;
  public static final int LOOP_INVARIANT = LBLNEG + 1;
  public static final int LOOP_PREDICATE = LOOP_INVARIANT + 1;
  public static final int LS = LOOP_PREDICATE + 1;
  public static final int MAIN = LS + 1; 
  public static final int MAX = MAIN + 1; // Function
  public static final int MODIFIES = MAX + 1;
  public static final int MONITORED = MODIFIES + 1;
  public static final int MONITORED_BY = MONITORED + 1;
  public static final int NON_NULL = MONITORED_BY + 1;
  public static final int NOWARN = NON_NULL + 1;
  public static final int OTI = NOWARN + 1;
  // "performs" keyword for procedure-modular reasoning
  public static final int PERFORMS = OTI + 1; 
  public static final int PRE = PERFORMS + 1;
  public static final int RES = PRE + 1;
  public static final int REQUIRES = RES + 1;
  public static final int SET = REQUIRES + 1;
  public static final int SKOLEM_CONSTANT = SET + 1;
  public static final int SPEC_PUBLIC = SKOLEM_CONSTANT + 1;
  public static final int STILL_DEFERRED = SPEC_PUBLIC + 1;
  public static final int THREAD_LOCAL = STILL_DEFERRED + 1;
  public static final int THREAD_LOCAL_RESULT = THREAD_LOCAL + 1;
  public static final int TI = THREAD_LOCAL_RESULT + 1;
  public static final int TID = TI + 1; 
  public static final int TYPE = TID + 1;	// "type"
  public static final int TYPETYPE = TYPE + 1;	  // "TYPE"; name for TYPECODE
  public static final int TYPEOF = TYPETYPE + 1; // Function
  public static final int UNINITIALIZED = TYPEOF + 1;
  public static final int UNREACHABLE = UNINITIALIZED + 1;
  public static final int WITNESS = UNREACHABLE + 1;
  public static final int WRITABLE_DEFERRED = WITNESS + 1;
  public static final int WRITABLE_IF = WRITABLE_DEFERRED+ 1;
  public static final int LASTESCKEYWORDTAG= WRITABLE_IF;

  //// Tags for ESC/Java checks
  public static final int FIRSTESCCHECKTAG = LASTESCKEYWORDTAG + 1;
  public static final int CHKARITHMETIC = FIRSTESCCHECKTAG;
  public static final int CHKARRAYSTORE = CHKARITHMETIC + 1;
  public static final int CHKASSERT = CHKARRAYSTORE + 1;
  public static final int CHKCLASSCAST= CHKASSERT + 1;
  public static final int CHKCODEREACHABILITY= CHKCLASSCAST + 1;
  public static final int CHKCONSTRUCTORLEAK= CHKCODEREACHABILITY + 1;
  public static final int CHKDEFINEDNESS= CHKCONSTRUCTORLEAK + 1;
  public static final int CHKELEMSGUARDED = CHKDEFINEDNESS + 1;
  public static final int CHKGLOBALINVARIANT= CHKELEMSGUARDED + 1;
  public static final int CHKGUARDED = CHKGLOBALINVARIANT + 1;
  public static final int CHKINDEXNEGATIVE= CHKGUARDED + 1;
  public static final int CHKINDEXTOOBIG = CHKINDEXNEGATIVE + 1;
  public static final int CHKINITIALIZATION= CHKINDEXTOOBIG + 1;
  public static final int CHKINITIALIZERLEAK= CHKINITIALIZATION + 1;
  public static final int CHKLEFTMOVER = CHKINITIALIZERLEAK + 1;  
  public static final int CHKLEFTMOVERENV = CHKLEFTMOVER + 1;  
  public static final int CHKLOCKINGORDER= CHKLEFTMOVERENV + 1;
  public static final int CHKLOOPINVARIANT= CHKLOCKINGORDER + 1;
  public static final int CHKLOOPOBJECTINVARIANT= CHKLOOPINVARIANT + 1;
  public static final int CHKMODIFIESEXTENSION= CHKLOOPOBJECTINVARIANT + 1;
  public static final int CHKMODIFIES= CHKMODIFIESEXTENSION + 1;
  public static final int CHKNEGATIVEARRAYSIZE= CHKMODIFIES + 1;
  public static final int CHKNONNULL= CHKNEGATIVEARRAYSIZE + 1;
  public static final int CHKNONNULLINIT= CHKNONNULL + 1;
  public static final int CHKNULLPOINTER= CHKNONNULLINIT + 1;
  public static final int CHKOBJECTINVARIANT= CHKNULLPOINTER + 1;
  public static final int CHKOWNERNULL= CHKOBJECTINVARIANT + 1;
  public static final int CHKOTI= CHKOWNERNULL + 1;
  public static final int CHKOTIREFL= CHKOTI + 1;
  public static final int CHKOTISTUTTER= CHKOTIREFL + 1;
  public static final int CHKOTITRANS= CHKOTISTUTTER + 1;
  public static final int CHKPERFORMS= CHKOTITRANS + 1;
  public static final int CHKPERFORMSEND= CHKPERFORMS + 1;
  public static final int CHKPOSTCONDITION= CHKPERFORMSEND + 1;
  public static final int CHKPRECONDITION= CHKPOSTCONDITION + 1;
  public static final int CHKRIGHTMOVER = CHKPRECONDITION + 1;  
  public static final int CHKRIGHTMOVERENV = CHKRIGHTMOVER + 1;  
  public static final int CHKSHARING= CHKRIGHTMOVERENV + 1;
  public static final int CHKUNENFORCEBLEOBJECTINVARIANT= CHKSHARING + 1;
  public static final int CHKUNEXPECTEDEXCEPTION= CHKUNENFORCEBLEOBJECTINVARIANT + 1;
  public static final int CHKWRITABLEDEFERRED= CHKUNEXPECTEDEXCEPTION + 1;
  public static final int CHKWRITABLE = CHKWRITABLEDEFERRED + 1;
  public static final int CHKFREE = CHKWRITABLE + 1;  
  public static final int LASTESCCHECKTAG = CHKFREE+1;

  //// Tags for Nary function symbols that are _not_ ESCJ keywords
  //
  // These need to be added both below and in escfunctions in this file,
  // as well as as switch labels in escjava.ast.EscPrettyPrint and
  // escjava.translate.VcToString.
  //
  public static final int FIRSTFUNCTIONTAG = LASTESCCHECKTAG;
  public static final int ALLOCLT = FIRSTFUNCTIONTAG;
  public static final int ALLOCLE = ALLOCLT+1;
  public static final int ANYEQ = ALLOCLE+1;
  public static final int ANYNE = ANYEQ+1;
  public static final int ARRAYLENGTH = ANYNE+1;
  public static final int ARRAYFRESH = ARRAYLENGTH + 1;
  public static final int ARRAYSHAPEMORE = ARRAYFRESH + 1;
  public static final int ARRAYSHAPEONE = ARRAYSHAPEMORE + 1;
  public static final int ASELEMS = ARRAYSHAPEONE + 1;
  public static final int ASFIELD = ASELEMS + 1;
  public static final int ASLOCKSET = ASFIELD + 1;
  public static final int BOOLAND = ASLOCKSET + 1;
  public static final int BOOLEQ = BOOLAND + 1;
  public static final int BOOLIMPLIES = BOOLEQ + 1;
  public static final int BOOLNE = BOOLIMPLIES + 1;
  public static final int BOOLNOT = BOOLNE + 1;
  public static final int BOOLOR = BOOLNOT + 1;
  public static final int CAST = BOOLOR + 1;
  public static final int CLASSLITERALFUNC = CAST + 1;
  public static final int CONDITIONAL = CLASSLITERALFUNC + 1;
  public static final int ECLOSEDTIME = CONDITIONAL + 1;
  // ELEMSNONNULL -- an ESC keyword
  // ELEMTYPE -- an ESC keyword
  public static final int FCLOSEDTIME = ECLOSEDTIME + 1;
  public static final int FLOATINGADD = FCLOSEDTIME + 1;
  public static final int FLOATINGDIV = FLOATINGADD + 1;
  public static final int FLOATINGEQ = FLOATINGDIV + 1;
  public static final int FLOATINGGE = FLOATINGEQ + 1;
  public static final int FLOATINGGT = FLOATINGGE + 1;
  public static final int FLOATINGLE = FLOATINGGT + 1;
  public static final int FLOATINGLT = FLOATINGLE + 1;
  public static final int FLOATINGMOD = FLOATINGLT + 1;
  public static final int FLOATINGMUL = FLOATINGMOD + 1;
  public static final int FLOATINGNE = FLOATINGMUL + 1;
  public static final int FLOATINGNEG = FLOATINGNE + 1;
  public static final int FLOATINGSUB = FLOATINGNEG + 1;
  public static final int INTEGRALADD = FLOATINGSUB + 1;
  public static final int INTEGRALAND = INTEGRALADD + 1;
  public static final int INTEGRALDIV = INTEGRALAND + 1;
  public static final int INTEGRALEQ = INTEGRALDIV + 1;
  public static final int INTEGRALGE = INTEGRALEQ + 1;
  public static final int INTEGRALGT = INTEGRALGE + 1;
  public static final int INTEGRALLE = INTEGRALGT + 1;
  public static final int INTEGRALLT = INTEGRALLE + 1;
  public static final int INTEGRALMOD = INTEGRALLT + 1;
  public static final int INTEGRALMUL = INTEGRALMOD + 1;
  public static final int INTEGRALNE = INTEGRALMUL + 1;
  public static final int INTEGRALNEG = INTEGRALNE + 1;
  public static final int INTEGRALNOT = INTEGRALNEG + 1;
  public static final int INTEGRALOR = INTEGRALNOT + 1;
  public static final int INTSHIFTL = INTEGRALOR + 1;
  public static final int LONGSHIFTL = INTSHIFTL + 1;
  public static final int INTSHIFTR = LONGSHIFTL + 1;
  public static final int LONGSHIFTR = INTSHIFTR + 1;
  public static final int INTSHIFTRU = LONGSHIFTR + 1;
  public static final int LONGSHIFTRU = INTSHIFTRU + 1;
  public static final int INTEGRALSUB = LONGSHIFTRU + 1;
  public static final int INTEGRALXOR = INTEGRALSUB + 1;
  public static final int IS = INTEGRALXOR + 1;
  public static final int ISALLOCATED = IS + 1;
  public static final int ISNEWARRAY = ISALLOCATED + 1;
  // MAX -- an ESC keyword
  public static final int LOCKLE = ISNEWARRAY + 1;
  public static final int LOCKLT = LOCKLE + 1;
  public static final int REFEQ = LOCKLT + 1;
  public static final int REFNE = REFEQ + 1;
  public static final int SELECT = REFNE + 1;
  public static final int STORE = SELECT + 1;
  public static final int STRINGCAT = STORE + 1;
  public static final int TYPEEQ = STRINGCAT + 1; 
  public static final int TYPENE = TYPEEQ + 1; 
  public static final int TYPELE = TYPENE + 1; // a.k.a. "<:"
  // TYPEOF -- an ESC keyword
  public static final int VALLOCTIME = TYPELE + 1;
  public static final int LASTFUNCTIONTAG = VALLOCTIME;

  // Constants used in deciding how to translate CHKs
  public static final int CHK_AS_ASSUME = LASTFUNCTIONTAG + 1;
  public static final int CHK_AS_ASSERT = CHK_AS_ASSUME + 1;
  public static final int CHK_AS_SKIP = CHK_AS_ASSERT + 1;
  public static final int LAST_TAG = CHK_AS_SKIP + 1;

  public static final Identifier ExsuresIdnName = Identifier.intern("Optional..Exsures..Id..Name");

  //// Helper functions

  public static String toVcString(int tag) {

    String final_string;
      
    switch(tag) {
    case IMPLIES:
      return "==>";
    case SUBTYPE:
      return "<:";
    case CHOICE:
      return "[]";
    case MAPSTO:
      return "->";
    case ANY:
      return "ANY";
    case TYPECODE:
      return "TYPECODE";		// displayed to user as "TYPE"
    case LOCKSET:
      return "LOCKSET";
    case CHK_AS_ASSUME:
      return "CHK_AS_ASSUME";
    case CHK_AS_ASSERT:
      return "CHK_AS_ASSERT";
    case CHK_AS_SKIP:
      return "CHK_AS_SKIP";
    default:
      if (FIRSTESCKEYWORDTAG <= tag && tag <= LASTESCKEYWORDTAG) 
	final_string = esckeywords[tag - FIRSTESCKEYWORDTAG].toString();
      else if (FIRSTESCCHECKTAG <= tag && tag < LASTESCCHECKTAG) 
	final_string = escchecks[tag - FIRSTESCCHECKTAG];
      else if (FIRSTFUNCTIONTAG <= tag && tag <= LASTFUNCTIONTAG) 
	final_string = escfunctions[tag - FIRSTFUNCTIONTAG];
      else if (tag < javafe.tc.TagConstants.LAST_TAG + 1) 
	final_string = javafe.tc.TagConstants.toString(tag);
      else 
	final_string = "Unknown ESC tag <" + tag
	    + " (+" + (tag-javafe.tc.TagConstants.LAST_TAG) + ") >";

      if (final_string.charAt(0) == '\\')
	// final_string = Atom.printableVersion(final_string);
	final_string = final_string.substring(1);
      if (final_string.equals("lockset"))
        final_string = "LS";
      else if (final_string.equals("result"))
        final_string = "RES";
      return final_string;
    }
  }


  public static String toString(int tag) {
    switch(tag) {
    case IMPLIES:
      return "==>";
    case SUBTYPE:
      return "<:";
    case MAPSTO:
      return "->";
    case ANY:
      return "ANY";
    case TYPECODE:
	//return "TYPECODE";		// displayed to user as "TYPE"
      return "TYPE";
    case LOCKSET:
      return "LOCKSET";
    case CHK_AS_ASSUME:
      return "CHK_AS_ASSUME";
    case CHK_AS_ASSERT:
      return "CHK_AS_ASSERT";
    case CHK_AS_SKIP:
      return "CHK_AS_SKIP";
    default:
      if (FIRSTESCKEYWORDTAG <= tag && tag <= LASTESCKEYWORDTAG)
	return esckeywords[tag - FIRSTESCKEYWORDTAG].toString();
      else if (FIRSTESCCHECKTAG <= tag && tag < LASTESCCHECKTAG)
	return escchecks[tag - FIRSTESCCHECKTAG];
      else if (FIRSTFUNCTIONTAG <= tag && tag <= LASTFUNCTIONTAG)
	return escfunctions[tag - FIRSTFUNCTIONTAG];
      else if (tag < javafe.tc.TagConstants.LAST_TAG + 1)
	return javafe.tc.TagConstants.toString(tag);
      else {
	return "Unknown ESC tag <" + tag
	    + " (+" + (tag-javafe.tc.TagConstants.LAST_TAG) + ") >";
      }
    }
  }

  public static int fromIdentifier(Identifier esckeyword) {
    for(int i = 0; i < esckeywords.length; i++)
      if (esckeyword == esckeywords[i]) return i + FIRSTESCKEYWORDTAG;
    return NULL;
  }

  public static int checkFromString(String s) {
    for (int i = FIRSTESCCHECKTAG; i <= LASTESCCHECKTAG; i++) {
      if (s.equals(escchecks[i - FIRSTESCCHECKTAG]))
	return i;
    }
    //@ unreachable
    Assert.fail("unrecognized check string: \"" + s + "\"");
    return -1;
  }
  
  private static Identifier[] esckeywords = {
    Identifier.intern("action"),
    Identifier.intern("also_ensures"),
    Identifier.intern("also_exsures"),
    Identifier.intern("also_modifies"),
    Identifier.intern("also_requires"),
    Identifier.intern("assume"),
    Identifier.intern("assert"),
    Identifier.intern("axiom"),
    Identifier.intern("readable_if"),
    Identifier.intern("\\dttfsa"),
    Identifier.intern("ensures"),
    Identifier.intern("elems_unwritable_by_env_if"),
    Identifier.intern("\\nonnullelements"),
    Identifier.intern("\\elemtype"),
    Identifier.intern("\\exists"),
    Identifier.intern("exsures"),
    Identifier.intern("\\fresh"),
    Identifier.intern("\\forall"),
    Identifier.intern("ghost"),
    Identifier.intern("global_invariant"),
    Identifier.intern("unwritable_by_env_if"),
    Identifier.intern("helper"),
    Identifier.intern("invariant"),
    Identifier.intern("\\lblpos"),
    Identifier.intern("\\lblneg"),
    Identifier.intern("loop_invariant"),
    Identifier.intern("loop_predicate"),
    Identifier.intern("\\lockset"),
    Identifier.intern("\\main"),
    Identifier.intern("\\max"),
    Identifier.intern("modifies"),
    Identifier.intern("monitored"),
    Identifier.intern("monitored_by"),
    Identifier.intern("non_null"),
    Identifier.intern("nowarn"),
    Identifier.intern("env_assumption"),
    Identifier.intern("performs"),
    Identifier.intern("\\old"),
    Identifier.intern("\\result"),
    Identifier.intern("requires"),
    Identifier.intern("set"),
    Identifier.intern("skolem_constant"),
    Identifier.intern("spec_public"),
    Identifier.intern("still_deferred"),
    Identifier.intern("thread_local"),
    Identifier.intern("thread_local_result"),
    Identifier.intern("ti"),
    Identifier.intern("\\tid"),
    Identifier.intern("\\type"),			// TYPE
    Identifier.intern("\\TYPE"),			// TYPETYPE
    Identifier.intern("\\typeof"),
    Identifier.intern("uninitialized"),
    Identifier.intern("unreachable"),
    Identifier.intern("\\witness"),
    Identifier.intern("writable_deferred"),
    Identifier.intern("writable_if")
  };

  private static String[] escchecks = {
    "ZeroDiv",
    "ArrayStore",
    "Assert",
    "Cast",
    "Reachable",
    "CLeak",
    "Unreadable",
    "ElemsGuarded",
    "GlobalInvariant",
    "Guarded",
    "IndexNegative",
    "IndexTooBig",
    "Uninit",
    "ILeak",
    "LeftMover",
    "LeftMoverEnv",
    "Deadlock",
    "LoopInv",
    "LoopObjInv",
    "ModExt",
    "Modifies",
    "NegSize",
    "NonNull",
    "NonNullInit",
    "Null",
    "Invariant",
    "OwnerNull",
    "OTI",
    "OTIReflexivity",
    "OTIStutter",
    "OTITransitivity",
    "CheckPerforms",
    "CheckPerformsEnd",
    "Post",
    "Pre",
    "RightMover",
    "RightMoverEnv",
    "Race",
    "Unenforcable",
    "Exception",
    "Deferred",
    "Writable",
    "Free"  // printed in debugging output only
  };

  private static String[] escfunctions = {
    "allocLT",
    "allocLE",
    "anyEQ",
    "anyNE",
    "arrayLength",
    "arrayFresh",
    "arrayShapeMore",
    "arrayShapeOne",
    "asElems",
    "asField",
    "asLockSet",
    "boolAnd",
    "boolEq",
    "boolImplies",
    "boolNE",
    "boolNot",
    "boolOr",
    "cast",
    "classLiteral",
    "termConditional",
    "eClosedTime",
    "fClosedTime",
    "floatingAdd",
    "floatingDiv",
    "floatingEQ",
    "floatingGE",
    "floatingGT",
    "floatingLE",
    "floatingLT",
    "floatingMod",
    "floatingMul",
    "floatingNE",
    "floatingNeg",
    "floatingSub",
    "integralAdd",
    "integralAnd",
    "integralDiv",
    "integralEQ",
    "integralGE",
    "integralGT",
    "integralLE",
    "integralLT",
    "integralMod",
    "integralMul",
    "integralNE",
    "integralNeg",
    "integralNot",
    "integralOr",
     "intShiftL",
     "longShiftL",
     "intShiftAR",
     "longShiftAR",
     "intShiftUR",
     "longShiftUR",
    "integralSub",
    "integralXor",
    "is",
    "isAllocated",
    "isNewArray",
    "lockLE",
    "lockLT",
    "refEQ",
    "refNE",
    "select",
    "store",
    "stringCat",
    "typeEQ",
    "typeNE",
    "typeLE",
    "vAllocTime"
  };

  static {
    Assert.notFalse( esckeywords.length ==
		     LASTESCKEYWORDTAG - FIRSTESCKEYWORDTAG +1 );
    Assert.notFalse( escchecks.length ==
		     LASTESCCHECKTAG - FIRSTESCCHECKTAG );
    Assert.notFalse( escfunctions.length ==
		     LASTFUNCTIONTAG - FIRSTFUNCTIONTAG +1 );
  }

  public static void main(String[] args) {
    for(int i= FIRST_TAG; i<LAST_TAG; i++ )
      System.out.println(i+" "+toString(i));
  }
}

