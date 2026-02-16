/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.ast;

import javafe.ast.Identifier;
import javafe.util.Assert;

import escjava.prover.Atom;

public class TagConstants extends GeneratedTags 
	//javafe.tc.TagConstants
    //implements GeneratedTags
{
    //// Tags for new binary operators
    public static final int IMPLIES = escjava.ast.GeneratedTags.LAST_TAG + 1;
    public static final int EXPLIES = IMPLIES + 1;
    public static final int IFF = EXPLIES + 1;  // equivalence (equality)
    public static final int NIFF = IFF + 1;     // discrepance (xor)
    public static final int SUBTYPE = NIFF + 1;
    public static final int DOTDOT = SUBTYPE + 1;

    //// Tags for pragma punctuation
    public static final int LEFTARROW = DOTDOT + 1; // <-
    public static final int RIGHTARROW = LEFTARROW + 1; // ->
    public static final int OPENPRAGMA = RIGHTARROW + 1; // {|
    public static final int CLOSEPRAGMA = OPENPRAGMA + 1; // |}

    //// Tags for new literal expressions
    public static final int SYMBOLLIT = CLOSEPRAGMA + 1;

    //// Tags for new primitive types
    public static final int ANY = SYMBOLLIT + 1;
    public static final int TYPECODE = ANY + 1;
    public static final int BIGINTTYPE = TYPECODE + 1;
    public static final int REALTYPE = BIGINTTYPE + 1;
    public static final int LOCKSET = REALTYPE + 1;
    public static final int OBJECTSET = LOCKSET + 1;
    
    //// Tags for guarded commands
    public static final int ASSERTCMD = OBJECTSET + 1;
    public static final int ASSUMECMD = ASSERTCMD + 1;
    public static final int CHOOSECMD = ASSUMECMD + 1;
    public static final int RAISECMD = CHOOSECMD + 1;
    public static final int SKIPCMD = RAISECMD + 1;
    public static final int TRYCMD = SKIPCMD + 1;

    //// Tags for special tokens
    public static final int INFORMALPRED_TOKEN = TRYCMD + 1;

    //// Tags for ESC/Java checks
    public static final int FIRSTESCCHECKTAG = INFORMALPRED_TOKEN + 1;
    public static final int CHKARITHMETIC = FIRSTESCCHECKTAG;
    public static final int CHKARRAYSTORE = CHKARITHMETIC + 1;
    public static final int CHKASSERT = CHKARRAYSTORE + 1;
    public static final int CHKCLASSCAST = CHKASSERT + 1;
    public static final int CHKCODEREACHABILITY = CHKCLASSCAST + 1;
    public static final int CHKCONSISTENT = CHKCODEREACHABILITY + 1;
    public static final int CHKCONSTRAINT = CHKCONSISTENT + 1;
    public static final int CHKCONSTRUCTORLEAK = CHKCONSTRAINT + 1;
    public static final int CHKDECREASES_BOUND = CHKCONSTRUCTORLEAK + 1;
    public static final int CHKDECREASES_DECR = CHKDECREASES_BOUND + 1;
    public static final int CHKDEFINEDNESS = CHKDECREASES_DECR + 1;
    public static final int CHKEXPRDEFINEDNESS = CHKDEFINEDNESS + 1;
    public static final int CHKEXPRDEFNORMPOST = CHKEXPRDEFINEDNESS + 1;
    public static final int CHKEXPRDEFEXCEPOST = CHKEXPRDEFNORMPOST + 1;
    public static final int CHKINDEXNEGATIVE = CHKEXPRDEFEXCEPOST + 1;
  //public static final int CHKINDEXNEGATIVE = CHKEXPRDEFINEDNESS + 1;
    public static final int CHKINDEXTOOBIG = CHKINDEXNEGATIVE + 1;
    public static final int CHKINITIALIZATION = CHKINDEXTOOBIG + 1;
    public static final int CHKINITIALIZERLEAK = CHKINITIALIZATION + 1;
    public static final int CHKINITIALLY = CHKINITIALIZERLEAK + 1;
    public static final int CHKLOCKINGORDER = CHKINITIALLY + 1;
    public static final int CHKLOOPINVARIANT = CHKLOCKINGORDER + 1;
    public static final int CHKLOOPOBJECTINVARIANT = CHKLOOPINVARIANT + 1;
    public static final int CHKMODIFIESEXTENSION = CHKLOOPOBJECTINVARIANT + 1;
    public static final int CHKMODIFIES = CHKMODIFIESEXTENSION + 1;
    public static final int CHKNEGATIVEARRAYSIZE = CHKMODIFIES + 1;
    public static final int CHKNONNULL = CHKNEGATIVEARRAYSIZE + 1;
    public static final int CHKNONNULLINIT = CHKNONNULL + 1;
    public static final int CHKNONNULLRESULT = CHKNONNULLINIT + 1;
    public static final int CHKNULLPOINTER = CHKNONNULLRESULT + 1;
    public static final int CHKOBJECTINVARIANT = CHKNULLPOINTER + 1;
    public static final int CHKOWNERNULL = CHKOBJECTINVARIANT + 1;
    public static final int CHKPOSTCONDITION = CHKOWNERNULL + 1;
    public static final int CHKPRECONDITION = CHKPOSTCONDITION + 1;
    public static final int CHKSHARING = CHKPRECONDITION + 1;
    public static final int CHKSHARINGALLNULL = CHKSHARING + 1;
    public static final int CHKUNENFORCEBLEOBJECTINVARIANT = CHKSHARINGALLNULL + 1;
    public static final int CHKUNEXPECTEDEXCEPTION = CHKUNENFORCEBLEOBJECTINVARIANT + 1;
    public static final int CHKUNEXPECTEDEXCEPTION2 = CHKUNEXPECTEDEXCEPTION + 1;
    public static final int CHKWRITABLEDEFERRED = CHKUNEXPECTEDEXCEPTION2 + 1;
    public static final int CHKWRITABLE = CHKWRITABLEDEFERRED + 1;
    public static final int CHKQUIET = CHKWRITABLE + 1;
    public static final int CHKASSUME = CHKQUIET + 1;
    public static final int CHKADDINFO = CHKASSUME + 1;
    public static final int CHKFREE = CHKADDINFO + 1;
    public static final int LASTESCCHECKTAG = CHKFREE;

    //// Tags for Nary function symbols that are _not_ ESCJ keywords
    //
    // These need to be added both below and in escfunctions in this file,
    // as well as as switch labels in escjava.ast.EscPrettyPrint and
    // escjava.translate.VcToString.
    //
    public static final int FIRSTFUNCTIONTAG = LASTESCCHECKTAG + 1;
    public static final int ALLOCLT = FIRSTFUNCTIONTAG;
    public static final int ALLOCLE = ALLOCLT+1;
    public static final int ANYEQ = ALLOCLE+1;
    public static final int ANYNE = ANYEQ+1;
    public static final int ARRAYLENGTH = ANYNE+1;
    public static final int ARRAYFRESH = ARRAYLENGTH + 1;
    public static final int ARRAYMAKE = ARRAYFRESH + 1;
    public static final int ARRAYSHAPEMORE = ARRAYMAKE + 1;
    public static final int ARRAYSHAPEONE = ARRAYSHAPEMORE + 1;
    public static final int ASELEMS = ARRAYSHAPEONE + 1;
    public static final int ASFIELD = ASELEMS + 1;
    public static final int ASLOCKSET = ASFIELD + 1;
    public static final int BOOLAND = ASLOCKSET + 1;
    public static final int BOOLANDX = BOOLAND + 1;
    public static final int BOOLEQ = BOOLANDX + 1;
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
    public static final int INTERN = INTEGRALXOR + 1;
    public static final int INTERNED = INTERN + 1;
    public static final int IS = INTERNED + 1;
    public static final int ISALLOCATED = IS + 1;
    public static final int ISNEWARRAY = ISALLOCATED + 1;
    // MAX -- an ESC keyword
    public static final int LOCKLE = ISNEWARRAY + 1;
    public static final int LOCKLT = LOCKLE + 1;
    public static final int METHODCALL = LOCKLT + 1;
    public static final int REFEQ = METHODCALL + 1;
    public static final int REFNE = REFEQ + 1;
    public static final int SELECT = REFNE + 1;
    public static final int STORE = SELECT + 1;
    public static final int STRINGCAT = STORE + 1;
    public static final int STRINGCATP = STRINGCAT + 1;
    public static final int TYPEEQ = STRINGCATP + 1; 
    public static final int TYPENE = TYPEEQ + 1; 
    public static final int TYPELE = TYPENE + 1; // a.k.a. "<:"
    // TYPEOF -- an ESC keyword
    public static final int UNSET = TYPELE + 1;
    public static final int VALLOCTIME = UNSET + 1;
    public static final int LASTFUNCTIONTAG = VALLOCTIME;

    // Constants used in deciding how to translate CHKs
    public static final int CHK_AS_ASSUME = LASTFUNCTIONTAG + 1;
    public static final int CHK_AS_ASSERT = CHK_AS_ASSUME + 1;
    public static final int CHK_AS_SKIP = CHK_AS_ASSERT + 1;

// FIXME - these should be merged into one order so they are easy to find
// Also keywords are looked up by a linear search - that could be improved
// upon greatly
    //// JML keywords
    public static final int FIRSTJMLKEYWORDTAG = CHK_AS_SKIP + 1;

    public static final int ASSUME = FIRSTJMLKEYWORDTAG;
    public static final int AXIOM = ASSUME + 1;
    public static final int CODE_CONTRACT = AXIOM + 1;
    public static final int DECREASES = CODE_CONTRACT + 1;
    public static final int DTTFSA = DECREASES + 1;
    public static final int ENSURES = DTTFSA + 1;
    public static final int ELEMSNONNULL = ENSURES + 1; // Function
    public static final int ELEMTYPE = ELEMSNONNULL + 1; // Function
    public static final int EXISTS = ELEMTYPE + 1;
    public static final int EXSURES = EXISTS + 1;
    public static final int FRESH = EXSURES + 1; // Non-GCE function
    public static final int FORALL = FRESH + 1;
    public static final int FUNCTION = FORALL + 1;
    public static final int GHOST = FUNCTION + 1;
    public static final int HELPER = GHOST + 1;
    public static final int IMMUTABLE = HELPER + 1;
    public static final int IN = IMMUTABLE + 1;
    public static final int IN_REDUNDANTLY = IN + 1;
    public static final int INTO = IN_REDUNDANTLY + 1;
    public static final int INVARIANT = INTO + 1;
    public static final int LBLPOS = INVARIANT + 1;
    public static final int LBLNEG = LBLPOS + 1;
    public static final int LOOP_INVARIANT = LBLNEG + 1;
    public static final int LOOP_PREDICATE = LOOP_INVARIANT + 1;
    public static final int LS = LOOP_PREDICATE + 1;
    public static final int MAPS = LS + 1; 
    public static final int MAPS_REDUNDANTLY = MAPS + 1; 
    public static final int MAX = MAPS_REDUNDANTLY + 1; // Function
    public static final int MODIFIES = MAX + 1;
    public static final int MONITORED = MODIFIES + 1;
    public static final int MONITORED_BY = MONITORED + 1;
    public static final int MONITORS_FOR = MONITORED_BY + 1;
    public static final int NON_NULL = MONITORS_FOR + 1;
    public static final int NOWARN = NON_NULL + 1;
    public static final int PRE = NOWARN + 1;
    public static final int READABLE = PRE + 1;
    public static final int READABLE_IF = READABLE + 1;
    public static final int RES = READABLE_IF + 1;
    public static final int REQUIRES = RES + 1;
    public static final int SET = REQUIRES + 1;
    public static final int SPEC_PUBLIC = SET + 1;
    public static final int STILL_DEFERRED = SPEC_PUBLIC + 1;
    public static final int TYPE = STILL_DEFERRED + 1;	// "type"
    public static final int TYPETYPE = TYPE + 1;	  // "TYPE"; name for TYPECODE
    public static final int TYPEOF = TYPETYPE + 1; // Function
    public static final int UNINITIALIZED = TYPEOF + 1;
    public static final int UNREACHABLE = UNINITIALIZED + 1;
    public static final int WRITABLE_DEFERRED = UNREACHABLE + 1;
    public static final int WRITABLE_IF = WRITABLE_DEFERRED+ 1;
    public static final int WRITABLE = WRITABLE_IF + 1;
    public static final int SKOLEM_CONSTANT = WRITABLE + 1;

    public static final int BIGINT = SKOLEM_CONSTANT + 1;
    public static final int WACK_DURATION = BIGINT + 1;
    // \elemtype -- an ESC keyword
    public static final int EVERYTHING = WACK_DURATION + 1;
    // \exists -- an ESC keyword
    public static final int FIELDS_OF = EVERYTHING + 1;
    // \forall -- an ESC keyword
    // \fresh -- an ESC keyword 
    public static final int INVARIANT_FOR = FIELDS_OF + 1;
    public static final int IS_INITIALIZED = INVARIANT_FOR + 1;
    // \lblneg -- an ESC keyword
    // \lblpos -- an ESC keyword
    // \lockset -- an ESC keyword
    // \max -- an ESC keyword 
    public static final int MAXQUANT = IS_INITIALIZED + 1;
    public static final int MIN = MAXQUANT + 1;
    // \nonnullelements -- an ESC keyword
    public static final int NOTHING = MIN + 1;
    public static final int NOT_MODIFIED = NOTHING + 1;
    public static final int NOT_SPECIFIED = NOT_MODIFIED + 1;
    public static final int WACK_NOWARN = NOT_SPECIFIED + 1;
    public static final int NOWARN_OP = WACK_NOWARN + 1;
    public static final int NUM_OF = NOWARN_OP + 1;
    // \old -- an ESC keyword
    public static final int OTHER = NUM_OF + 1;
    public static final int PRIVATE_DATA = OTHER + 1;
    public static final int PRODUCT = PRIVATE_DATA + 1;
    public static final int REACH = PRODUCT + 1;
    public static final int REAL = REACH + 1;
    // \result -- an ESC keyword
    public static final int SPACE = REAL + 1;
    public static final int SUCH_THAT = SPACE + 1;
    public static final int SUM = SUCH_THAT + 1;
    // \type -- an ESC keyword
    // \typeof -- an ESC keyword
    // \TYPE -- an ESC keyword
    public static final int WARN = SUM + 1;
    public static final int WARN_OP = WARN + 1;
    public static final int WACK_WORKING_SPACE = WARN_OP + 1;

    public static final int ABRUPT_BEHAVIOR = WACK_WORKING_SPACE + 1;
    public static final int ACCESSIBLE_REDUNDANTLY = ABRUPT_BEHAVIOR + 1;
    public static final int ACCESSIBLE = ACCESSIBLE_REDUNDANTLY + 1;
    public static final int ALSO = ACCESSIBLE + 1;
    public static final int ALSO_REFINE = ALSO + 1; // Internal use only
    public static final int ASSERT_REDUNDANTLY = ALSO_REFINE + 1;
    public static final int ASSIGNABLE_REDUNDANTLY = ASSERT_REDUNDANTLY + 1;
    public static final int ASSIGNABLE = ASSIGNABLE_REDUNDANTLY + 1;
    public static final int ONLY_ASSIGNED = ASSIGNABLE + 1;
    public static final int ASSUME_REDUNDANTLY = ONLY_ASSIGNED + 1;
    // assume -- an ESC keyword
    // axiom -- an ESC keyword
    public static final int BEHAVIOR = ASSUME_REDUNDANTLY + 1;
    public static final int BREAKS_REDUNDANTLY = BEHAVIOR + 1;
    public static final int BREAKS = BREAKS_REDUNDANTLY + 1;
    public static final int CALLABLE_REDUNDANTLY = BREAKS + 1;
    public static final int CALLABLE = CALLABLE_REDUNDANTLY + 1;
    public static final int CHOOSE_IF = CALLABLE + 1;
    public static final int CHOOSE = CHOOSE_IF + 1;
    public static final int CONSTRAINT_REDUNDANTLY = CHOOSE + 1;
    public static final int CONSTRAINT = CONSTRAINT_REDUNDANTLY + 1;
    public static final int CONSTRUCTOR = CONSTRAINT + 1;
    public static final int CONTINUES_REDUNDANTLY = CONSTRUCTOR + 1;
    public static final int CONTINUES = CONTINUES_REDUNDANTLY + 1; // @review
    public static final int DECREASES_REDUNDANTLY = CONTINUES + 1;
    // decreases -- an ESC keyword
    public static final int DECREASING_REDUNDANTLY = DECREASES_REDUNDANTLY + 1;
    public static final int DECREASING = DECREASING_REDUNDANTLY + 1;
    public static final int DEPENDS_REDUNDANTLY = DECREASING + 1;
    public static final int DEPENDS = DEPENDS_REDUNDANTLY + 1;
    public static final int DIVERGES_REDUNDANTLY = DEPENDS + 1;
    public static final int DIVERGES = DIVERGES_REDUNDANTLY + 1;
    public static final int DURATION_REDUNDANTLY = DIVERGES + 1;
    public static final int DURATION = DURATION_REDUNDANTLY + 1;
    public static final int END = DURATION + 1; // Internal use only
    public static final int ENSURES_REDUNDANTLY = END + 1;
    // ensures -- an ESC keyword
    public static final int EXAMPLE = ENSURES_REDUNDANTLY + 1;
    public static final int EXCEPTIONAL_BEHAVIOR = EXAMPLE + 1;
    public static final int EXCEPTIONAL_EXAMPLE = EXCEPTIONAL_BEHAVIOR + 1;
    public static final int EXSURES_REDUNDANTLY = EXCEPTIONAL_EXAMPLE + 1;
    public static final int FIELDKW = EXSURES_REDUNDANTLY + 1;
    // exsures -- an ESC keyword
    public static final int NO_WACK_FORALL = FIELDKW + 1;
    public static final int FOR_EXAMPLE = NO_WACK_FORALL + 1;
    // ghost -- an ESC keyword
    public static final int IMPLIES_THAT = FOR_EXAMPLE + 1;
    // helper -- an ESC keyword
    public static final int HENCE_BY_REDUNDANTLY = IMPLIES_THAT + 1;
    public static final int HENCE_BY = HENCE_BY_REDUNDANTLY + 1;
    public static final int INITIALIZER = HENCE_BY + 1;
    public static final int INITIALLY = INITIALIZER + 1;
    public static final int INSTANCE = INITIALLY + 1;
    public static final int INVARIANT_REDUNDANTLY = INSTANCE + 1;
    // invariant -- an ESC keyword
    public static final int LOOP_INVARIANT_REDUNDANTLY = INVARIANT_REDUNDANTLY + 1;
    // loop_invariant -- an ESC keyword
    public static final int MAINTAINING_REDUNDANTLY = LOOP_INVARIANT_REDUNDANTLY + 1;
    public static final int MAINTAINING = MAINTAINING_REDUNDANTLY + 1;
    public static final int MEASURED_BY_REDUNDANTLY = MAINTAINING + 1;
    public static final int MEASURED_BY = MEASURED_BY_REDUNDANTLY + 1;
    public static final int METHOD = MEASURED_BY + 1;
    public static final int MODEL = METHOD + 1;
    public static final int MODEL_PROGRAM = MODEL + 1;
    public static final int MODIFIABLE_REDUNDANTLY = MODEL_PROGRAM + 1;
    public static final int MODIFIABLE = MODIFIABLE_REDUNDANTLY + 1;
    public static final int MODIFIES_REDUNDANTLY = MODIFIABLE + 1;
    // modifies -- an ESC keyword
    // monitored_by -- an ESC keyword
    // monitored -- an ESC keyword
    public static final int NESTEDMODIFIERPRAGMA = MODIFIES_REDUNDANTLY + 1;
    // non_null -- an ESC keyword
    public static final int NORMAL_BEHAVIOR = NESTEDMODIFIERPRAGMA + 1;
    public static final int NORMAL_EXAMPLE = NORMAL_BEHAVIOR + 1;
    // nowarn -- an ESC keyword
    public static final int OLD = NORMAL_EXAMPLE + 1;
    public static final int MODELPROGRAM_OR = OLD + 1;
    public static final int PARSEDSPECS = MODELPROGRAM_OR + 1;
    public static final int POSTCONDITION_REDUNDANTLY = PARSEDSPECS + 1;
    public static final int POSTCONDITION = POSTCONDITION_REDUNDANTLY + 1;
    public static final int PRECONDITION_REDUNDANTLY = POSTCONDITION + 1;
    public static final int PRECONDITION = PRECONDITION_REDUNDANTLY + 1;
    public static final int PURE = PRECONDITION + 1;
    // readable_if -- an ESC keyword
    public static final int SAME = PURE + 1;
    public static final int REFINE = SAME + 1;
    public static final int REFINES = REFINE + 1;
    public static final int REPRESENTS_REDUNDANTLY = REFINES + 1;
    public static final int REPRESENTS = REPRESENTS_REDUNDANTLY + 1;
    public static final int REQUIRES_REDUNDANTLY = REPRESENTS + 1;
    // requires -- an ESC keyword
    public static final int RETURNS_REDUNDANTLY = REQUIRES_REDUNDANTLY + 1;
    public static final int RETURNS = RETURNS_REDUNDANTLY + 1;
    // set -- an ESC keyword
    public static final int SIGNALS_REDUNDANTLY = RETURNS + 1;
    public static final int SIGNALS = SIGNALS_REDUNDANTLY + 1;
    public static final int SIGNALS_ONLY = SIGNALS + 1;
    public static final int SPEC_PROTECTED = SIGNALS_ONLY + 1;
    // spec_public -- an ESC keyword
    public static final int STATIC_INITIALIZER = SPEC_PROTECTED + 1;
    public static final int SUBCLASSING_CONTRACT = STATIC_INITIALIZER + 1;
    // uninitialized -- an ESC keyword
    // unreachable -- an ESC keyword
    public static final int WEAKLY = SUBCLASSING_CONTRACT + 1;
    public static final int WHEN_REDUNDANTLY = WEAKLY + 1;
    public static final int WHEN = WHEN_REDUNDANTLY + 1;
    public static final int WORKING_SPACE_REDUNDANTLY = WHEN + 1;
    public static final int WORKING_SPACE = WORKING_SPACE_REDUNDANTLY + 1;

    public static final int LASTJMLKEYWORDTAG = WORKING_SPACE;

    //// Tags for ESCJ keywords
    // These are keywords that are not in JML (either obsolete or
    // extensions), or are tokens for internal use only
    // Be sure to keep the esckeywords[] array in synch
    public static final int FIRSTESCKEYWORDTAG = LASTJMLKEYWORDTAG + 1;
    public static final int ALSO_ENSURES = FIRSTESCKEYWORDTAG;
    public static final int ALSO_EXSURES = ALSO_ENSURES + 1;
    public static final int ALSO_MODIFIES = ALSO_EXSURES + 1;
    public static final int ALSO_REQUIRES = ALSO_MODIFIES + 1;

    // Include the Universe type annotation keywords (cjbooms)
    //alx: dw
    public static final int PEER = ALSO_REQUIRES + 1;
    public static final int READONLY = PEER + 1;
    public static final int REP = READONLY + 1;
    //alx-end

    // Chalin-Kiniry experimental keywords for nullness, purity, and whatnot
    public static final int NULLABLE = REP + 1;
    public static final int NULLABLE_BY_DEFAULT = NULLABLE + 1;
    public static final int NON_NULL_BY_DEFAULT = NULLABLE_BY_DEFAULT + 1;
    public static final int OBS_PURE = NON_NULL_BY_DEFAULT + 1;

    // Chalin's spec and code modifiers for different math semantics
    public static final int WACK_JAVA_MATH = OBS_PURE + 1;
    public static final int WACK_SAFE_MATH = WACK_JAVA_MATH + 1;
    public static final int WACK_BIGINT_MATH = WACK_SAFE_MATH + 1;
    public static final int CODE_JAVA_MATH = WACK_BIGINT_MATH + 1;
    public static final int CODE_SAFE_MATH = CODE_JAVA_MATH + 1;
    public static final int CODE_BIGINT_MATH = CODE_SAFE_MATH + 1;
    public static final int SPEC_JAVA_MATH = CODE_BIGINT_MATH + 1;
    public static final int SPEC_SAFE_MATH = SPEC_JAVA_MATH + 1;
    public static final int SPEC_BIGINT_MATH = SPEC_SAFE_MATH + 1;

    public static final int LASTESCKEYWORDTAG = SPEC_BIGINT_MATH;

    public static final int LAST_TAG = LASTESCKEYWORDTAG;

    public static final /*@non_null*/ Identifier ExsuresIdnName = 
        Identifier.intern("Optional..Exsures..Id..Name");

    //// Helper functions

    public static /*@non_null*/ String toVcString(int tag) {
        switch(tag) {
            case TYPECODE:
                return "TYPECODE";		// displayed to user as "TYPE"
            default:
                break;
        }

        String s = toString(tag);
        //@ assume s.length() > 0;
        if (s.charAt(0) == '\\') {
            s = s.substring(1);
        }
        if (s.equals("lockset")) {
            s = "LS";
        } else if (s.equals("result")) {
            s = "RES";
        }
        return s;
    }

    // Documented in parent.

    public static /*@non_null*/ String toString(int tag) {
        switch(tag) {
            // new literal expression (not true keyword)
            case  SYMBOLLIT:
                return "*SYMBOLLIT*";
            // guarded commands (not true keywords)
            case ASSERTCMD:
                return "*ASSERTCMD*";
            case ASSUMECMD:
                return "*ASSUMECMD*";
            case CHOOSECMD:
                return "*CHOOSECMD*";
            case RAISECMD:
                return "*RAISECMD*";
            case SKIPCMD:
                return "*SKIPCMD*";
            case TRYCMD:
                return "*TRYCMD*";
            // informal predicates (not true keyword)
            case INFORMALPRED_TOKEN:
                return "*INFORMAL_PREDICATE*";
            // special symbols
            case IMPLIES:
                return "==>";
            case EXPLIES:
                return "<==";
            case IFF:
                return "<==>";
            case NIFF:
                return "<=!=>";
            case SUBTYPE:
                return "<:";
	    case DOTDOT:
		return "..";
	    case LEFTARROW:
		return "<-";
	    case RIGHTARROW:
		return "->";
	    case OPENPRAGMA:
		return "{|";
	    case CLOSEPRAGMA:
		return "|}";
            case ANY:
                return "ANY";
            case TYPECODE:
                //return "TYPECODE";		// displayed to user as "TYPE"
                return "TYPE";
            case LOCKSET:
                return "LOCKSET";
            case OBJECTSET:
                return "OBJECTSET";
            case CHK_AS_ASSUME:
                return "CHK_AS_ASSUME";
            case CHK_AS_ASSERT:
                return "CHK_AS_ASSERT";
            case CHK_AS_SKIP:
                return "CHK_AS_SKIP";
            default:
                if (FIRSTESCKEYWORDTAG <= tag && tag <= LASTESCKEYWORDTAG)
                    return esckeywords[tag - FIRSTESCKEYWORDTAG].toString();
                else if (FIRSTESCCHECKTAG <= tag && tag <= LASTESCCHECKTAG)
                    return escchecks[tag - FIRSTESCCHECKTAG];
                else if (FIRSTFUNCTIONTAG <= tag && tag <= LASTFUNCTIONTAG)
                    return escfunctions[tag - FIRSTFUNCTIONTAG];
                else if (FIRSTJMLKEYWORDTAG <= tag && tag <= LASTJMLKEYWORDTAG)
                    return jmlkeywords[tag - FIRSTJMLKEYWORDTAG].toString();
		else if (tag == TagConstants.MODIFIESGROUPPRAGMA)
		    return "modifies";
                else if (tag <= GeneratedTags.LAST_TAG)
                    return GeneratedTags.toString(tag);
                else {
                    return "Unknown ESC tag <" + tag
                        + " (+" + (tag - javafe.tc.TagConstants.LAST_TAG) + ") >";
                }
        }
    }

    /**
     * @param keyword the keyword to lookup.
     * @return the index of the {@link TagConstants} attribute
     * corresponding to the keyword encoded as an {@link Identifier}
     * in the parameter 'keyword'. A {@link #NULL} is returned if
     * the identifier in question is not an ESC/Java or JML keyword
     * known to {@link TagConstants}.
     */
    public static int fromIdentifier(Identifier keyword) {
        for(int i = 0; i < jmlkeywords.length; i++)
            if (keyword == jmlkeywords[i]) return i + FIRSTJMLKEYWORDTAG;
        for(int i = 0; i < esckeywords.length; i++)
            if (keyword == esckeywords[i]) return i + FIRSTESCKEYWORDTAG;
        return NULL;
    }

    public static boolean isKeywordTag(int tag) {
	return (FIRSTJMLKEYWORDTAG <= tag && tag <= LASTJMLKEYWORDTAG)
          || (FIRSTESCKEYWORDTAG <= tag && tag <= LASTESCKEYWORDTAG)
          || tag == TagConstants.NULLLIT;
    }

    public static int checkFromString(/*@non_null*/ String s) {
        for (int i = FIRSTESCCHECKTAG; i <= LASTESCCHECKTAG; i++) {
            if (s.equals(escchecks[i - FIRSTESCCHECKTAG]))
                return i;
        }
        //@ unreachable;
        Assert.fail("unrecognized check string: \"" + s + "\"");
        return -1;
    }

    /**
     * @return a "redundant-fied" version of the passed tag if it can
     * be made redundant; otherwise, return the parameter unchanged.
     */
    public static int makeRedundant(int tag) {
        int Result = tag;
        switch (tag) {
	    case TagConstants.REQUIRES:
                Result = TagConstants.REQUIRES_REDUNDANTLY; break;
            case TagConstants.ENSURES:
                Result = TagConstants.ENSURES_REDUNDANTLY; break;
            case TagConstants.PRECONDITION:
                Result = TagConstants.PRECONDITION_REDUNDANTLY; break;
            case TagConstants.DIVERGES:
                Result = TagConstants.DIVERGES_REDUNDANTLY; break;
            case TagConstants.WHEN:
                Result = TagConstants.WHEN_REDUNDANTLY; break;
            case TagConstants.POSTCONDITION:
                Result = TagConstants.POSTCONDITION_REDUNDANTLY; break;
            case TagConstants.EXSURES:
                Result = TagConstants.EXSURES_REDUNDANTLY; break;
            case TagConstants.SIGNALS:
                Result = TagConstants.SIGNALS_REDUNDANTLY; break;
            case TagConstants.MODIFIABLE:
                Result = TagConstants.MODIFIABLE_REDUNDANTLY; break;
            case TagConstants.ASSIGNABLE:
                Result = TagConstants.ASSIGNABLE_REDUNDANTLY; break;
            case TagConstants.MODIFIES:
                Result = TagConstants.MODIFIES_REDUNDANTLY; break;
            case TagConstants.MEASURED_BY:
                Result = TagConstants.MEASURED_BY_REDUNDANTLY; break;
            case TagConstants.ASSERT:
                Result = TagConstants.ASSERT_REDUNDANTLY; break;
            case TagConstants.ASSUME:
                Result = TagConstants.ASSUME_REDUNDANTLY; break;
            case TagConstants.LOOP_INVARIANT:
                Result = TagConstants.LOOP_INVARIANT_REDUNDANTLY; break;
            case TagConstants.MAINTAINING:
                Result = TagConstants.MAINTAINING_REDUNDANTLY; break;
            case TagConstants.DECREASES:
                Result = TagConstants.DECREASES_REDUNDANTLY; break;
            case TagConstants.INVARIANT:
                Result = TagConstants.INVARIANT_REDUNDANTLY; break;
            case TagConstants.REPRESENTS:
                Result = TagConstants.REPRESENTS_REDUNDANTLY; break;
            case TagConstants.CONSTRAINT:
                Result = TagConstants.CONSTRAINT_REDUNDANTLY; break;
            case TagConstants.DECREASING:
                Result = TagConstants.DECREASING_REDUNDANTLY; break;
            case TagConstants.DURATION:
                Result = TagConstants.DURATION_REDUNDANTLY; break;
            case TagConstants.WORKING_SPACE:
                Result = TagConstants.WORKING_SPACE_REDUNDANTLY; break;
            case TagConstants.IN:
                Result = TagConstants.IN_REDUNDANTLY; break;
            case TagConstants.MAPS:
                Result = TagConstants.MAPS_REDUNDANTLY; break;
            case TagConstants.HENCE_BY:
                Result = TagConstants.HENCE_BY_REDUNDANTLY; break;
        }
        return Result;
    }
    
    /**
     * @return an "unredundant-fied" version of the passed tag if it
     * is redundant; otherwise, return the parameter unchanged.
     */
    public static int unRedundant(int tag) {
        int Result = tag;
        switch (tag) {
	          case TagConstants.REQUIRES_REDUNDANTLY:
                Result = TagConstants.REQUIRES; break;
            case TagConstants.ENSURES_REDUNDANTLY:
                Result = TagConstants.ENSURES; break;
            case TagConstants.PRECONDITION_REDUNDANTLY:
                Result = TagConstants.PRECONDITION; break;
            case TagConstants.DIVERGES_REDUNDANTLY:
                Result = TagConstants.DIVERGES; break;
            case TagConstants.WHEN_REDUNDANTLY:
                Result = TagConstants.WHEN; break;
            case TagConstants.POSTCONDITION_REDUNDANTLY:
                Result = TagConstants.POSTCONDITION; break;
            case TagConstants.EXSURES_REDUNDANTLY:
                Result = TagConstants.EXSURES; break;
            case TagConstants.SIGNALS_REDUNDANTLY:
                Result = TagConstants.SIGNALS; break;
            case TagConstants.MODIFIABLE_REDUNDANTLY:
                Result = TagConstants.MODIFIABLE; break;
            case TagConstants.ASSIGNABLE_REDUNDANTLY:
                Result = TagConstants.ASSIGNABLE; break;
            case TagConstants.MODIFIES_REDUNDANTLY:
                Result = TagConstants.MODIFIES; break;
            case TagConstants.MEASURED_BY_REDUNDANTLY:
                Result = TagConstants.MEASURED_BY; break;
            case TagConstants.ASSERT_REDUNDANTLY:
                Result = TagConstants.ASSERT; break;
            case TagConstants.ASSUME_REDUNDANTLY:
                Result = TagConstants.ASSUME; break;
            case TagConstants.LOOP_INVARIANT_REDUNDANTLY:
                Result = TagConstants.LOOP_INVARIANT; break;
            case TagConstants.MAINTAINING_REDUNDANTLY:
                Result = TagConstants.MAINTAINING; break;
            case TagConstants.DECREASES_REDUNDANTLY:
                Result = TagConstants.DECREASES; break;
            case TagConstants.INVARIANT_REDUNDANTLY:
                Result = TagConstants.INVARIANT; break;
            case TagConstants.REPRESENTS_REDUNDANTLY:
                Result = TagConstants.REPRESENTS; break;
            case TagConstants.CONSTRAINT_REDUNDANTLY:
                Result = TagConstants.CONSTRAINT; break;
            case TagConstants.DECREASING_REDUNDANTLY:
                Result = TagConstants.DECREASING; break;
            case TagConstants.DURATION_REDUNDANTLY:
                Result = TagConstants.DURATION; break;
            case TagConstants.WORKING_SPACE_REDUNDANTLY:
                Result = TagConstants.WORKING_SPACE; break;
            case TagConstants.IN_REDUNDANTLY:
                Result = TagConstants.IN; break;
            case TagConstants.MAPS_REDUNDANTLY:
                Result = TagConstants.MAPS; break;
            case TagConstants.HENCE_BY_REDUNDANTLY:
                Result = TagConstants.HENCE_BY; break;
        }
        return Result;
    }

    public static boolean isRedundant(int tag) {
      return (tag == TagConstants.REQUIRES_REDUNDANTLY
              || tag == TagConstants.ASSERT_REDUNDANTLY
              || tag == TagConstants.ASSIGNABLE_REDUNDANTLY
              || tag == TagConstants.ASSUME_REDUNDANTLY
              || tag == TagConstants.CONSTRAINT_REDUNDANTLY
              || tag == TagConstants.DECREASES_REDUNDANTLY
              || tag == TagConstants.DECREASING_REDUNDANTLY
              || tag == TagConstants.DIVERGES_REDUNDANTLY
              || tag == TagConstants.DURATION_REDUNDANTLY
              || tag == TagConstants.ENSURES_REDUNDANTLY
              || tag == TagConstants.EXSURES_REDUNDANTLY
              || tag == TagConstants.HENCE_BY_REDUNDANTLY
              || tag == TagConstants.INVARIANT_REDUNDANTLY
              || tag == TagConstants.IN_REDUNDANTLY
              || tag == TagConstants.LOOP_INVARIANT_REDUNDANTLY
              || tag == TagConstants.MAINTAINING_REDUNDANTLY
              || tag == TagConstants.MAPS_REDUNDANTLY
              || tag == TagConstants.MEASURED_BY_REDUNDANTLY
              || tag == TagConstants.MODIFIABLE_REDUNDANTLY
              || tag == TagConstants.MODIFIES_REDUNDANTLY
              || tag == TagConstants.POSTCONDITION_REDUNDANTLY
              || tag == TagConstants.PRECONDITION_REDUNDANTLY
              || tag == TagConstants.REPRESENTS_REDUNDANTLY
              || tag == TagConstants.SIGNALS_REDUNDANTLY
              || tag == TagConstants.WHEN_REDUNDANTLY
              || tag == TagConstants.WORKING_SPACE_REDUNDANTLY
              );
    }

    public final static /*@non_null*/ String[] escchecks = {
        "ZeroDiv",
        "ArrayStore",
        "Assert",
        "Cast",
        "Reachable",
	"Inconsistent",
	"Constraint",
        "CLeak",
        "DecreasesBound",
        "Decreases",
        "Unreadable",
	"Undefined",
	"UndefinedNormalPost",
	"UndefinedExceptionalPost",
        "IndexNegative",
        "IndexTooBig",
        "Uninit",
        "ILeak",
	"Initially",
        "Deadlock",
        "LoopInv",
        "LoopObjInv",
        "ModExt",
        "Modifies",
        "NegSize",
        "NonNull",
        "NonNullInit",
        "NonNullResult",
        "Null",
        "Invariant",
        "OwnerNull",
        "Post",
        "Pre",
        "Race",
	"RaceAllNull",
        "Unenforcable",
        "Exception",
        "SpecificationException",
        "Deferred",
        "Writable",
        "vc.Quiet",  // printed in debugging output only
	"Assume",  // internal use only
	"AdditionalInfo", // internal use only
        "Free"  // printed in debugging output only
    };
        
    private static /*@non_null*/ String[] escfunctions = {
        "allocLT",
        "allocLE",
        "anyEQ",
        "anyNE",
        "arrayLength",
        "arrayFresh",
        "arrayMake",
        "arrayShapeMore",
        "arrayShapeOne",
        "asElems",
        "asField",
        "asLockSet",
        "boolAnd",
        "boolAndX",
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
        "|intern:|", // maps int to String for Strings interned by ESC/Java
        "|interned:|", // maps String to boolean to say whether a String is interned, per Java semantics
        "is",
        "isAllocated",
        "isNewArray",
        "lockLE",
        "lockLT",
	"methodCall",
        "refEQ",
        "refNE",
        "select",
        "store",
        "stringCat",
        "stringCatP",
        "typeEQ",
        "typeNE",
        "typeLE",
        "unset",
        "vAllocTime"
    };

    static public final String STRINGCATINFIX = "java.lang.String._infixConcat_";
				     // Must match method name in String.spec

    private static /*@non_null*/ Identifier[] jmlkeywords = {
        Identifier.intern("assume"),
        Identifier.intern("axiom"),
        Identifier.intern("code_contract"),
        Identifier.intern("decreases"),
        Identifier.intern("\\dttfsa"),
        Identifier.intern("ensures"),
        Identifier.intern("\\nonnullelements"),
        Identifier.intern("\\elemtype"),
        Identifier.intern("\\exists"),
        Identifier.intern("exsures"),
        Identifier.intern("\\fresh"),
        Identifier.intern("\\forall"),
	Identifier.intern("function"),
        Identifier.intern("ghost"),
        Identifier.intern("helper"),
        Identifier.intern("immutable"),
        Identifier.intern("in"),
        Identifier.intern("in_redundantly"),
        Identifier.intern("\\into"),
        Identifier.intern("invariant"),
        Identifier.intern("\\lblpos"),
        Identifier.intern("\\lblneg"),
        Identifier.intern("loop_invariant"),
        Identifier.intern("loop_predicate"),
        Identifier.intern("\\lockset"),
	Identifier.intern("maps"),
	Identifier.intern("maps_redundantly"),
        Identifier.intern("\\max"),
        Identifier.intern("modifies"),
        Identifier.intern("monitored"),
        Identifier.intern("monitored_by"),
        Identifier.intern("monitors_for"),
        Identifier.intern("non_null"),
        Identifier.intern("nowarn"),
        Identifier.intern("\\old"),  // TagConstants.PRE
        Identifier.intern("readable"),
        Identifier.intern("readable_if"),
        Identifier.intern("\\result"),
        Identifier.intern("requires"),
        Identifier.intern("set"),
        Identifier.intern("spec_public"),
        Identifier.intern("still_deferred"),
        Identifier.intern("\\type"),			// TYPE
        Identifier.intern("\\TYPE"),			// TYPETYPE
        Identifier.intern("\\typeof"),
        Identifier.intern("uninitialized"),
        Identifier.intern("unreachable"),
        Identifier.intern("writable_deferred"),
        Identifier.intern("writable_if"),
        Identifier.intern("writable"),
        Identifier.intern("skolem_constant"),
        Identifier.intern("\\bigint"),
        Identifier.intern("\\duration"),
        Identifier.intern("\\everything"),
        Identifier.intern("\\fields_of"),
        Identifier.intern("\\invariant_for"),
        Identifier.intern("\\is_initialized"),
        Identifier.intern("\\max"),
        Identifier.intern("\\min"),
        Identifier.intern("\\nothing"),
        Identifier.intern("\\not_modified"),
        Identifier.intern("\\not_specified"),
	Identifier.intern("\\nowarn"),
	Identifier.intern("\\nowarn_op"),
        Identifier.intern("\\num_of"),
        Identifier.intern("\\other"),
        Identifier.intern("\\private_data"),
        Identifier.intern("\\product"),
        Identifier.intern("\\reach"),
        Identifier.intern("\\real"),
        Identifier.intern("\\space"),
        Identifier.intern("\\such_that"),
        Identifier.intern("\\sum"),
	Identifier.intern("\\warn"),
	Identifier.intern("\\warn_op"),
        Identifier.intern("\\working_space"),
        Identifier.intern("abrupt_behavior"),
        Identifier.intern("accessible_redundantly"),
        Identifier.intern("accessible"),
        Identifier.intern("also"),
	Identifier.intern("---also_refine---"), // internal use only
        Identifier.intern("assert_redundantly"),
        Identifier.intern("assignable_redundantly"),
        Identifier.intern("assignable"),
        Identifier.intern("\\only_assigned"),
        Identifier.intern("assume_redundantly"),
        Identifier.intern("behavior"),
        Identifier.intern("breaks_redundantly"),
        Identifier.intern("breaks"),
        Identifier.intern("callable_redundantly"),
        Identifier.intern("callable"),
        Identifier.intern("choose_if"),
        Identifier.intern("choose"),
        Identifier.intern("constraint_redundantly"),
        Identifier.intern("constraint"),
	Identifier.intern("constructor"),
        Identifier.intern("continues_redundantly"),
        Identifier.intern("continues"),
        Identifier.intern("decreases_redundantly"),
        Identifier.intern("decreasing_redundantly"),
        Identifier.intern("decreasing"),
        Identifier.intern("depends_redundantly"),
        Identifier.intern("depends"),
        Identifier.intern("diverges_redundantly"),
        Identifier.intern("diverges"),
        Identifier.intern("duration_redundantly"),
        Identifier.intern("duration"),
	Identifier.intern("---end---"), // Internal use only
        Identifier.intern("ensures_redundantly"),
        Identifier.intern("example"),
        Identifier.intern("exceptional_behavior"),
        Identifier.intern("exceptional_example"),
        Identifier.intern("exsures_redundantly"),
        Identifier.intern("field"),
        Identifier.intern("forall"),
        Identifier.intern("for_example"),
        Identifier.intern("implies_that"),
        Identifier.intern("hence_by_redundantly"),
        Identifier.intern("hence_by"),
        Identifier.intern("initializer"),
        Identifier.intern("initially"),
        Identifier.intern("instance"),
        Identifier.intern("invariant_redundantly"),
        Identifier.intern("loop_invariant_redundantly"),
        Identifier.intern("maintaining_redundantly"),
        Identifier.intern("maintaining"),
        Identifier.intern("measured_by_redundantly"),
        Identifier.intern("measured_by"),
        Identifier.intern("method"),
        Identifier.intern("model"),
        Identifier.intern("model_program"),
        Identifier.intern("modifiable_redundantly"),
        Identifier.intern("modifiable"),
        Identifier.intern("modifies_redundantly"),
	Identifier.intern("--- nested specs ---"),
        Identifier.intern("normal_behavior"),
        Identifier.intern("normal_example"),
        Identifier.intern("old"),
        Identifier.intern("or"),
	Identifier.intern("--- parsed specs ---"),
        Identifier.intern("post_redundantly"),
        Identifier.intern("post"),
        Identifier.intern("pre_redundantly"),
        Identifier.intern("pre"), // PRE not PRE (which is \old )
        Identifier.intern("pure"),
        Identifier.intern("\\same"),
        Identifier.intern("refine"),
        Identifier.intern("refines"),
        Identifier.intern("represents_redundantly"),
        Identifier.intern("represents"),
        Identifier.intern("requires_redundantly"),
        Identifier.intern("returns_redundantly"),
        Identifier.intern("returns"),
        Identifier.intern("signals_redundantly"),
        Identifier.intern("signals"),
        Identifier.intern("signals_only"),
        Identifier.intern("spec_protected"),
        Identifier.intern("static_initializer"),
        Identifier.intern("subclassing_contract"),
        Identifier.intern("weakly"),
        Identifier.intern("when_redundantly"),
        Identifier.intern("when"),
        Identifier.intern("working_space_redundantly"),
        Identifier.intern("working_space")
    };

    private static /*@non_null*/ Identifier[] esckeywords = {
        Identifier.intern("also_ensures"),
        Identifier.intern("also_exsures"),
        Identifier.intern("also_modifies"),
        Identifier.intern("also_requires"),

        // Universe type annotations (cjbooms)
 	      Identifier.intern("\\peer"),
        Identifier.intern("\\readonly"),
        Identifier.intern("\\rep"),

        // Chalin-Kiniry experimental keywords for nullness, purity, and whatnot
        Identifier.intern("nullable"),
        Identifier.intern("nullable_by_default"),
        Identifier.intern("non_null_by_default"),
        Identifier.intern("obs_pure"),

        // Chalin's spec and code modifiers for different math semantics
        Identifier.intern("\\java_math"),
        Identifier.intern("\\safe_math"),
        Identifier.intern("\\bigint_math"),
        Identifier.intern("code_java_math"),
        Identifier.intern("code_safe_math"),
        Identifier.intern("code_bigint_math"),
        Identifier.intern("spec_java_math"),
        Identifier.intern("spec_safe_math"),
        Identifier.intern("spec_bigint_math")
    };

    public static void main(/*@ non_null @*/ String[] args) {
        for(int i = FIRST_TAG; i <= LAST_TAG; i++ )
            System.out.println(i + " " + toString(i));
    }

    // This initialization code is simply a quick check that the arrays
    // are consistent in length.

    static private void comp(int i, int j, /*@non_null*/ String s) {
	if (i != j)
		System.out.println("Mismatched length ("
			+ i + " vs. " + j + ") in " + s);
    }

    static {
	comp(esckeywords.length, LASTESCKEYWORDTAG - FIRSTESCKEYWORDTAG + 1,
		"esckeywords");
	comp(jmlkeywords.length, LASTJMLKEYWORDTAG - FIRSTJMLKEYWORDTAG + 1,
		"jmlkeywords");
	comp(escfunctions.length, LASTFUNCTIONTAG - FIRSTFUNCTIONTAG + 1,
		"escfunctions");
	comp(escchecks.length, LASTESCCHECKTAG - FIRSTESCCHECKTAG + 1,
		"escchecks");
    }
}
