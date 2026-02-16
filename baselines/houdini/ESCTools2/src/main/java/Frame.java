/*
 * Created on Jan 8, 2005
 *
 */
package escjava.translate;

import java.util.Hashtable;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;

import javafe.ast.ASTNode;
import javafe.ast.ArrayRefExpr;
import javafe.ast.Expr;
import javafe.ast.ExprObjectDesignator;
import javafe.ast.ExprVec;
import javafe.ast.FieldAccess;
import javafe.ast.FieldDecl;
import javafe.ast.Identifier;
import javafe.ast.LiteralExpr;
import javafe.ast.ObjectDesignator;
import javafe.ast.RoutineDecl;
import javafe.ast.SuperObjectDesignator;
import javafe.ast.ThisExpr;
import javafe.ast.Type;
import javafe.ast.TypeDecl;
import javafe.ast.TypeObjectDesignator;
import javafe.ast.VariableAccess;
import javafe.tc.EnvForTypeSig;
import javafe.tc.TypeSig;
import javafe.util.ErrorSet;
import javafe.util.Location;
import escjava.Main;
import escjava.ast.ArrayRangeRefExpr;
import escjava.ast.CondExprModifierPragmaVec;
import escjava.ast.EverythingExpr;
import escjava.ast.Modifiers;
import escjava.ast.ModifiesGroupPragma;
import escjava.ast.ModifiesGroupPragmaVec;
import escjava.ast.NaryExpr;
import escjava.ast.NothingExpr;
import escjava.ast.TagConstants;
import escjava.ast.WildRefExpr;
import escjava.tc.Datagroups;
import escjava.tc.FlowInsensitiveChecks;
import escjava.tc.TypeCheck;
import escjava.tc.Types;


/**
 * @author David Cok
 *
 * This class contains methods for handling frame conditions.
 * <p>
 * Note the following about the implementation of Modifies Pragmas.
 * The desugared frame specifications of a method are contained in a
 * ModifiesGroupPragmaVec, which is a vector of ModifiesGroupPragma elements.
 * Each ModifiesGroupPragma instance corresponds to the frame conditions for
 * a single specification case.  Since each specification case of a method
 * must be independently satisfied, each ModifiesGroupPragma must be 
 * satisfied.
 * <p>
 * A ModifiesGroupPragma contains a precondition (the Expr <obj>.precondition)
 * and a set of CondExprModifierPragma elements in a CondExprModifierPragmaVec.
 * Each CondExprModifierPragma corresponds to one store-ref location as might
 * be listed in an assignable clause (e.g. a field, an array reference, 
 * wild-card items such as obj.* or array[*], array ranges such as array[i..j]).
 * The Expr that is in a CondExprModifierPragma is obsolete. [It used to be
 * that an assignable clause had an optional conditional expression, but that
 * has been deprecated.] The list of store-refs in a ModifierGroupPragma is
 * essentially a union - within a specification case a specifier can list 
 * multiple store-ref locations that may be assigned.  A particular assignment
 * statement will have a lhs which must match one of the store-ref locations
 * in the ModifiesGroup Pragma, and must do so for each ModifiesGroupPragma
 * in the ModifiesGroupPragmaVec.
 * <p>
 * 
 */
public class Frame {
  
  /** The Translate instance that owns this instance of Frame */
  final private Translate translator;
  
  /** The value of issueCautions obtained from the Translate owner */
  final private boolean issueCautions;
  
  /** The RoutineDecl whose body is in the process of being translated */
  final private RoutineDecl rdCurrent;
  
  /** The mapping to be used for \old variables to get pre-state values */
  final private Hashtable premap;
  
  /** This holds a value across recursive invocations of methods
   * within this class; it designates that the object whose field
   * is being assigned is known to be fresh.
   */
  private boolean pFreshResult = false;
  
  /** A precomputed Identifier for 'this', to be used in constructing
   *  Expressions.
   */
  static private final Identifier thisId = Identifier.intern("this");
  
  /** The constructor of a Frame instance; should be called only from
   *  Translate
   * @param t The instance of Translate to which this Frame belongs
   * @param issueCautions Whether to issue Caution messages to the user
   * @param rdCurrent The RoutineDecl whose body is being translated
   * @param premap The mapping to pre-state values to be used in translation
   */
  public Frame(Translate t, boolean issueCautions, RoutineDecl rdCurrent,
      Hashtable premap) {
    translator = t; 
    this.issueCautions = issueCautions;
    this.rdCurrent = rdCurrent;
    this.premap = premap;
  }
  
  // These fields are used in the course of generating checks
  
  /** The kind of expression being checked, e.g. assignment or method call.
   *  Used in the message displayed to the user when the check fails.
   */
  private String kindOfModCheck = "assignment";
  
  /** This method returns whether the given field (fd) of the given object
   * (eod) (which is null if the field is static) is definitely not allowed
   * to be assigned according to the specs of the current RoutineDecl.  By
   * definitely not allowed, we mean that it can be determined without needing
   * to prove the truth or falsity of a given expression.  For example, if
   * the FieldDecl is nowhere mentioned in the ModifiesGroupPragma, then it
   * definitely may not be assigned.  In a situation where the assignable
   * clause contained something like <expr>.field where field matches fd,
   * then the assignment is allowed if <expr> == eod.  But that would have
   * to be proved: it might be that the assignment is not allowed, if 
   * <expr> != eod, but we don't definitely know for sure.
   * @param eod The object whose field is being assigned, or null if the field
   *    is static
   * @param fd The declaration corresponding to the field being assigned.
   * @return true if it is known without a prover that the field may not be
   *    assigned; false if it may or possibly may be assigned
   */
  public boolean isDefinitelyNotAssignable(Expr eod, FieldDecl fd) {
    definitelyNotAssignable = true;
    modifiesCheckFieldHelper(eod, Location.NULL, fd,
        Location.NULL, null, false, null, null, null);
    boolean r = definitelyNotAssignable;
    definitelyNotAssignable = false;
    return r;
  }

  /** A private variable used to get information without using the
   * return value, in support of method isDefinitelylNotAssignable.
   */
  private boolean definitelyNotAssignable = false;
  
  /** This method generates checks that the field in lhs is allowed to be
   *  assigned according to the specification of the current RoutineDecl
   *  (rdCurrent).  This may depend on the object whose field is being
   *  assigned.  This is called for both static and instance fields.
   * @param lhs The expression being assigned to (which will be a field
   *              dereference).
   * @param callLoc The location of the field access expression (the lhs).
   * @param fd The location of the declaration of the field being assigned
   */
  void modifiesCheckField(Expr lhs, int callLoc, FieldDecl fd) {
    kindOfModCheck = "assignment";
    // FIXME - I don't think this handles this and super that are not the
    // prefix.
    if (!issueCautions) return;
    // eod is the object part of the FieldAccess, as in an assigment of the
    // form  <eod>.<field> = expr;
    Expr eod = null;
    if (lhs instanceof NaryExpr) {
      eod = (Expr)((NaryExpr)lhs).childAt(2);
    } else if (lhs instanceof VariableAccess) {
      // static field
      // eod stays null
    } else {
      ErrorSet.caution(callLoc,"INTERNAL ERROR: Unhandled type of lhs in Frame.modifiesCheckField "
          + lhs.getClass());
      escjava.ast.EscPrettyPrint.inst.println(System.out,lhs);
      return;
    }
    modifiesCheckFieldHelper(eod,callLoc,fd,Location.NULL,
        null,false,null,null,null);
  }
  
  /** This is a helper method to generate checks that a particular
   * field assignment is allowed by the frame conditions.
   * @param eod The object owning the field being assigned; null if the field
   *              is static
   * @param callLoc The location of the field access being assigned to
   * @param fd   The declaration of the field being assigned to
   * @param calleeLoc The location of the assignment or method call 
   * @param callee_tprecondition The precondition under which the callee might
   *          assign this field; null is equivalent to true
   * @param genExpr true if an Expr containing the formjula to be checked is
   *          to be returned, false if checks are to be inserted in the code
   * @param mg  the set of specifications against which the assignment of the
   *          field is to be tested
   * @param thisexpr the expression to be used for 'this' in translating
   *          input expressions, or null if no translation is to be done
   * @param args the set of variable mappings to be used in translation
   * @return the expression that must be true to allow the assignment, if 
   *    genExpr is true; if genExpr is false, null is returned
   */
  private Expr modifiesCheckFieldHelper(Expr eod, int callLoc,
      FieldDecl fd, int calleeLoc, 
      Expr callee_tprecondition, boolean genExpr,
      ModifiesGroupPragmaVec mg,
      Expr thisexpr, Hashtable args) {
    
    // We need to create a translated predicate expression that is true
    // if the lhs is allowed to be modified and false if it is not
    // allowed to be modified.  This will be an OR across each of the
    // modify pragmas.  Each pragma will contribute
    //   - obviously false if it is not the same declaration or same type
    //   - obviously true if is is the same declaration and both have 'this'
    //  as the od and the condition is true
    //   - obviously true (and then no check) if it is modifies everything
    //  with a true condition
    //   - some expression which is the AND of the condition and that
    //  the lhs matches the modifies expression:
    //      fields must have the same object designator
    //      arrays must have the same array expression and same index
    
    // Loop over each specification case in turn
    if (mg == null) mg = GetSpec.getCombinedMethodDecl(rdCurrent).modifies;
    ExprVec eev = ExprVec.make(mg.size());
    for (int kk=0; kk<mg.size(); ++kk) {
      ModifiesGroupPragma mge = mg.elementAt(kk); // The ModifiesGroupPragma
      // for the current specification case
      int callerLoc = mge.clauseLoc;  // Location of one of the modifies 
      // clauses for this specification case
      // ev collects assertions - if any one of them is true then the
      // assignment is allowed; if nothing is in it then it is equivalent
      // to 'false'; if ev is later set to null it means that a literal
      // 'true' would have been added, so the composite meaning is true
      ExprVec ev = ExprVec.make(10);

      // If the field is not a static field, then the assignment is
      // ok if the owning object is fresh since the pre-state.
      if (!definitelyNotAssignable && !genExpr && !Modifiers.isStatic(fd.modifiers)) addAllocExpression(ev,eod);
      // FIXME - why this check on definitelyNotAssignable????
      
      // Now iterate over all the store-refs in this specification case,
      // including store-refs that are in datagroups - we use this
      // iterator to hide the expansion of the datagroups
      ModifiesIterator caller_iterator = new ModifiesIterator(
                                          rdCurrent.parent,mge.items,true);
      while (ev != null && caller_iterator.hasNext()) {
        Object ex = caller_iterator.next();
        if (ex instanceof FieldAccess || ex instanceof FieldDecl) {
          FieldDecl fdd;
          ObjectDesignator odd;
          if (ex instanceof FieldAccess) {
            fdd = ((FieldAccess)ex).decl;
            odd = ((FieldAccess)ex).od;
          } else {
            fdd = (FieldDecl)ex;
            odd = null;
          }
          if (fd == fdd) {
            // The declaration for the assigned field is the same as the
            // declaration in the assignable clause
            if (Modifiers.isStatic(fd.modifiers)) {
              // The matching declarations are static - so they match.
              ev = null; 
            } else {
              // Instance references, so the objects need to be equal in order
              // for the store-refs to match
              Expr e1 = eod;
              Expr e2 = odd instanceof ExprObjectDesignator ?
                  ((ExprObjectDesignator)odd).expr:
                    ThisExpr.make(null,Location.NULL);
              if ( ((e1 instanceof ThisExpr) || 
                  ((e1 instanceof VariableAccess) && 
                      ((VariableAccess)e1).id == thisId)) && 
                      e2 instanceof ThisExpr) {
                // The objects references are both 'this'
                ev = null;
              } else {
                // Create a check that the two object references are equal
                ExprVec evv = ExprVec.make(1);
                evv.addElement(e2);
                e1 = eod;
                e2 = modTranslate(e2,!genExpr,thisexpr,args);
                Expr e = GC.nary(TagConstants.REFEQ,e1,e2);
                ev.addElement(e);
              }
            }
          }
        } else if (ex instanceof ArrayRefExpr) {
          // skip - not a match
        } else if (ex instanceof NothingExpr) {
          // skip - not a match
        } else if (ex instanceof EverythingExpr) {
          // anything matches \everything
          ev = null;
        } else if (ex instanceof ArrayRangeRefExpr) {
          // skip - not a match
        } else if (ex instanceof WildRefExpr) {
          ObjectDesignator odd = ((WildRefExpr)ex).od;
          if (odd instanceof TypeObjectDesignator) {
            if (Modifiers.isStatic(fd.modifiers)) {
              // A static field reference matches <Type>.* if
              // the static field is a field of the Type or its subtypes
              TypeSig t = TypeCheck.inst.getSig(fd.parent); // type in which the field is declared
              Type tt = ((TypeObjectDesignator)odd).type;
              if (t == tt || Types.isSubclassOf(tt,t)) {
                ev = null;
              }
            } // If not static then is not a match
          } else if (odd instanceof ExprObjectDesignator) {
            if (!Modifiers.isStatic(fd.modifiers)) {
              // An instance field matches <obj>.* if the 
              // object references are equal
              Expr e1 = eod;
              Expr e2 = modTranslate(((ExprObjectDesignator)odd).expr,
                  !genExpr,thisexpr,args);
              e1 = GC.nary(TagConstants.REFEQ,e1,e2);
              ev.addElement(e1);
            }  // If not an instance reference then not a match
          } else {
            ErrorSet.caution("INTERNAL ERROR: Unhandled ObjectDesignator case in modifiesCheckFieldHelper " + odd.getClass());
            escjava.ast.EscPrettyPrint.inst.println(System.out,odd);
          }
        } else {
          ErrorSet.caution("INTERNAL ERROR: Unhandled case in modifiesCheckFieldHelper " + ex.getClass());
          escjava.ast.EscPrettyPrint.inst.println(System.out,ex);
        }
      }
      // We have looped over all of the assignable store-refs in the
      // specification case.  If ev is null, there has been a definite
      // match and the assignment is ok for this case; if ev is empty
      // then there is no store-ref that is even a possible match;
      // otherwise the assignment is ok if the disjunction of the contents of ev
      // is true - and that needs to be proven by the checker.
      
      // Two conditions
      // if this method has been called on an assignment, we just have
      //    a callerLoc
      // if the method has been called on a method call, we have a
      //    calleeLoc and a callerLoc and we want the calleeLoc to be
      //    the first associated location

      // definitelyNotAssignable must be true on return if there is any 
      // specification case that does not allow the assignment
 
      if (definitelyNotAssignable) {
        //TrAnExpr.closeForClause();
        // FIXME - why don't we check the preconditions here
        if (ev != null && ev.size() == 0) return null;
        
      } else {
        
        Expr e = modChecksComplete(mge.precondition,
            callee_tprecondition,ev,callLoc,
            calleeLoc==Location.NULL ? callerLoc : calleeLoc,
                calleeLoc==Location.NULL ? Location.NULL: callerLoc, genExpr);
        if (genExpr && ev != null) eev.addElement(e);
      }
    }
    definitelyNotAssignable = false;
    return !genExpr ? null : eev.size() == 0 ? null : GC.and(eev);
  }
  
  /* Implementation note on frame conditions when methods are called.
   * 
   * We have frame conditions for the callee.  Any store-ref that
   * might be assigned by the callee at this point in the program
   * must be allowed to be assigned by the caller.  So we iterate
   * over each store-ref in each ModifiesGroupPragma in each
   * specification case of the callee, testing to see whether that
   * store-ref is allowed to be assigned by the caller.
   * <p>
   * There are a couple wrinkles.  First, there is a callee_precondition
   * that states under what circumstances the callee's store-ref might
   * possibly be assigned.  Only if that is true does the store-ref need
   * to be assignable by the caller.
   * <p>
   * Second, just because a store-ref is listed as assignable in a spec case
   * of the callee does not mean that the callee is actually allowed to assign
   * to it.  Such assignment may be precluded by another spec case of the callee.
   * For example, given
   * <pre>
         requires P;
         modifies i,j;
         also
         requires Q;
         modifies i,k;
     </pre>
   * j may be assigned only if P && !Q, since if Q is true only
   * i and k, but not j may be assigned.  Thus we have to test each store-ref
   * of the callee against all of the spec cases of its own specification.
   */
  
  /** This method generates checks that the locations possibly assigned by
   *  a called method are allowed to be assigned by the caller.
   * @param calleeFrameConditions the frame conditions of the callee
   * @param eod the receiver object of the call
   * @param loccall the location of the call
   * @param args the mapping of the arguments of the call
   * @param freshResult - true if eod is known to be fresh since the prestate
   * @param td_callee The TypeDecl in which the called method is declared
   */
  void modifiesCheckMethodI(ModifiesGroupPragmaVec calleeFrameConditions, 
      Expr eod, int loccall, Hashtable args, boolean freshResult,
      TypeDecl td_callee) {
    if (!issueCautions) return;
    kindOfModCheck = "method call";
    for (int i=0; i<calleeFrameConditions.size(); ++i) {
      ModifiesGroupPragma mg = calleeFrameConditions.elementAt(i);
      // FIXME - the precondition should not be null - guarding against bugs
      // upstream - e.g. current ArcType, but that may be because of model type problems
      if (mg.precondition == null) {
        //System.out.println("ADDING LIT " + Location.toString(mg.clauseLoc));
        mg.precondition = LiteralExpr.make(TagConstants.BOOLEANLIT, 
            Boolean.TRUE, Location.NULL);
        javafe.tc.FlowInsensitiveChecks.setType(mg.precondition,Types.booleanType);
      }
      Expr tp = modTranslate(mg.precondition,false,eod,args);
      ModifiesIterator callee_iterator = new ModifiesIterator(td_callee,mg.items,false);
      while (callee_iterator.hasNext()) {
        Object ex = callee_iterator.next();
        Expr e = modifiesCheckMethod(eod, Location.NULL, 
            args, false, 
            ex, 
            tp,true,calleeFrameConditions);
        if (e == GC.falselit) {
          continue;
        }
 
        ModifiesGroupPragmaVec mge = GetSpec.getCombinedMethodDecl(rdCurrent).modifies;
        modifiesCheckMethod(eod, loccall, 
            args, freshResult, 
            ex, 
            e == null ? tp : GC.and(tp,e),false,mge);
      }
    }
  }
  
  /**
   * Helper method that checks that a particular store-ref in
   * the frame conditions of a
   * called method against the frame conditions of the caller.
   * 
   * @param eod The receiver object of the called method
   * @param loccall  The location of the call
   * @param args  The mapping to be used for callee variables
   * @param freshResult true if the receiver is known to be fresh (allocated
   *          since the pre-state of the caller)
   * @param calleeStoreRef the store-ref of the callee to check
   * @param callee_tprecondition  The precondition of the store-ref under investigation
   * @param genExpr if true, an expression is returned that must be true to allow
   *    the store-ref; if false, a check is created and null is returned
   * @param mg the caller frame conditions against which to check
   * @return if genExpr is true, the expression that constitutes the condition
   *  that must be true if the frame condition is to be allowed is returned;
   *  if genExpr is false, null is returned (a assertion check is inserted into
   *  the generated guarded commands)
   */
  private Expr modifiesCheckMethod(Expr eod, int loccall, 
      Hashtable args, 
      boolean freshResult,
      Object calleeStoreRef,
      Expr callee_tprecondition, boolean genExpr,
      ModifiesGroupPragmaVec mg) {
    
    int calleeLoc = ((ASTNode)calleeStoreRef).getStartLoc();
    pFreshResult = freshResult;
    ExprVec eev = ExprVec.make(mg.size());
    try {
      // Iterating over specification cases
      for (int kk=0; kk<mg.size(); ++kk) {
        // Check that ex is assignable for each specification case
        // in the caller.  Need to issue an error if ex is not assignable;
        // need to issue a check to be proven if it might or might not be
        // assignable; can skip the iteration if ex is definitely assignable
        ModifiesGroupPragma mge = mg.elementAt(kk);
        int callerLoc = mge.getStartLoc();
        ExprVec ev = ExprVec.make(10);
        
        if (calleeStoreRef instanceof EverythingExpr) {
          ModifiesIterator caller_iterator = new ModifiesIterator(
                                               rdCurrent.parent,mge.items,true);
          while (caller_iterator.hasNext()) {
            Object callerStoreRef = caller_iterator.next();
            if (callerStoreRef instanceof EverythingExpr) {
              // calleeStoreRef is \everything but there is also an \everything in 
              // the caller, so there is a match
              ev = null;
              break;
            }
          }
        } else if (calleeStoreRef instanceof NothingExpr) {
          ev = null; // Anything in the caller will match
        } else if (calleeStoreRef instanceof FieldAccess) {
          FieldAccess fa = (FieldAccess)calleeStoreRef;
          Expr eeod = (fa.od instanceof ExprObjectDesignator) ? 
              ((ExprObjectDesignator)fa.od).expr : null;
          Expr lval = eeod == null ? null 
              : modTranslate(eeod, false,  eod, args);
          
          Expr e = modifiesCheckFieldHelper(lval,loccall, fa.decl, calleeLoc, 
              callee_tprecondition,genExpr,mg,eod,args);
          if (genExpr && e != null) eev.addElement(e);
          // A bit of a hack - the FieldHelper routine iterates over
          // all of the caller frame conditions, so we short-circuit
          // that here.  Skip the modChecksComplete at the end as well.
          break;
        } else if (calleeStoreRef instanceof ArrayRefExpr) {
          Expr array= modTranslate(((ArrayRefExpr)calleeStoreRef).array, false,  eod, args );
          Expr index= modTranslate(((ArrayRefExpr)calleeStoreRef).index, false,  eod, args );
          modifiesCheckArray(array,index,loccall,calleeLoc, callee_tprecondition,
              genExpr,mg,eod,args);
          // A bit of a hack - the helper routine iterates over
          // all of the caller frame conditions, so we short-circuit
          // that here.  Skip the modChecksComplete at the end as well.
          break;
        } else if (calleeStoreRef instanceof WildRefExpr) {
          ObjectDesignator odd = ((WildRefExpr)calleeStoreRef).od;
          Expr e1 = null;
          if (odd instanceof ExprObjectDesignator) {
            e1 = modTranslate(((ExprObjectDesignator)odd).expr,
                false,eod,args);
            if (!genExpr) addAllocExpression(ev,e1);
          }
          ModifiesIterator caller_iterator = new ModifiesIterator(
                                                 rdCurrent.parent,mge.items,true);
          while (caller_iterator.hasNext()) {
            Object callerStoreRef = caller_iterator.next();
            //System.out.println("CALLER " + callerStoreRef);
            if (callerStoreRef instanceof EverythingExpr) {
              ev = null;
              break;
            } else if (callerStoreRef instanceof WildRefExpr) {
              ObjectDesignator oddd = ((WildRefExpr)callerStoreRef).od;
              if ((odd instanceof TypeObjectDesignator) && (oddd instanceof TypeObjectDesignator)) {
                Type t = ((TypeObjectDesignator)odd).type;
                Type tt = ((TypeObjectDesignator)oddd).type;
                if (t == tt || ((t instanceof TypeSig) && Types.isSubclassOf(tt,(TypeSig)t)) ) {
                  ev = null;
                  break;
                }
              } else if (odd instanceof ExprObjectDesignator && oddd instanceof ExprObjectDesignator) {
                Expr e2 = modTranslate(((ExprObjectDesignator)oddd).expr,
                    !genExpr, eod, args);
                e1 = GC.nary(TagConstants.REFEQ,e1,e2);
                ev.addElement(e1);
              } else if (odd instanceof SuperObjectDesignator || oddd instanceof SuperObjectDesignator) {
                ErrorSet.caution("INTERNAL ERROR: Unhandled ObjectDesignator in Frame.modifiesCheckMethod: " + odd.getClass());
              }
            }
          }
        } else if (calleeStoreRef instanceof ArrayRangeRefExpr) {
          ArrayRangeRefExpr aexpr = (ArrayRangeRefExpr)calleeStoreRef;
          Expr array = aexpr.array;
          Expr lowIndex = aexpr.lowIndex;
          Expr highIndex = aexpr.highIndex;
          Expr ao = modTranslate(array,false,eod,args);
          Expr al = lowIndex == null ? null :
            modTranslate(lowIndex,false,eod,args); 
          Expr ah = highIndex == null ? null :
            modTranslate(highIndex,false,eod,args); 
          if (!genExpr) addAllocExpression(ev,ao);
          ModifiesIterator caller_iterator = new ModifiesIterator(
					  rdCurrent.parent,mge.items,true);
          while (caller_iterator.hasNext()) {
            Object callerStoreRef = caller_iterator.next();
            if (callerStoreRef instanceof EverythingExpr) {
              ev = null;
            } else if (callerStoreRef instanceof ArrayRangeRefExpr) {
              ArrayRangeRefExpr aaexpr = (ArrayRangeRefExpr)callerStoreRef;
              Expr aarray = aaexpr.array;
              Expr alowIndex = aaexpr.lowIndex;
              Expr ahighIndex = aaexpr.highIndex;
              Expr aao = modTranslate(aarray,!genExpr,eod,args);
              Expr aal = alowIndex == null ? null :
                modTranslate(alowIndex,!genExpr,eod,args); 
              Expr aah = ahighIndex == null ? null :
                modTranslate(ahighIndex,!genExpr,eod,args);
              // ao, aao are the callee/caller array expressions
              // al, aal are the callee/caller low index expressions, or
              //      null if the low index is not specified
              // ah, aah are the callee/caller high index expressions, or
              //      null if the high index is not specified
              // All expressions are already translated
              // We need to have the array expressions be equal
              // AND   aal <= al  AND ah <= aah
              // A null low index is equivalent to 0
              // A null high index is equivalent to the length-1
              // (since the arrays have to be the same, the lengths can be
              // presumed to be the same)
              if (ah == null && aah != null) continue; // FIXME - could compare against the length of ao
              aao = GC.nary(TagConstants.REFEQ,ao,aao);
              aal = aal == null ? null :
                GC.nary(TagConstants.INTEGRALLE,aal,
                    al != null ? al: GC.zerolit);
              aah = aah == null ? null :
                GC.nary(TagConstants.INTEGRALLE,ah,aah);
              aal = aal == null ? aah : aah == null ? aal : GC.and(aal,aah);
              aao = aal == null ? aao : GC.and(aao,aal);
              ev.addElement(aao);
            } else if (callerStoreRef instanceof ArrayRefExpr) {
              // Here the callee is an array range;
              // the caller is an array element; they match if
              // the range is just the single element.
              // FIXME: Note that we don't find matches of a
              // callee array range against the corresponding list
              // of individual array elements in the caller (or against
              // a list of caller array range store refs that when
              // combined match the callee)
              if (ah == null) continue; // FIXME - could compare against the length of ao
              ArrayRefExpr aaexpr = (ArrayRefExpr)callerStoreRef;
              Expr aarray = aaexpr.array;
              Expr aindex = aaexpr.index;
              Expr aao = modTranslate(aarray,!genExpr,eod,args);
              aindex = modTranslate(aindex,!genExpr,eod,args); 
              aao = GC.nary(TagConstants.REFEQ,ao,aao);
              Expr aal = GC.nary(TagConstants.INTEGRALLE,aindex,
                  al != null ? al: GC.zerolit);
              Expr aah = GC.nary(TagConstants.INTEGRALLE,ah,aindex);
              aal = GC.and(aal,aah);
              aao = GC.and(aao,aal);
              ev.addElement(aao);
            }
          }
        } else {
          // Leaving ev empty is equivalent to false
          ErrorSet.caution("INTERNAL ERROR: Modifies Check not implemented for " + calleeStoreRef.getClass());
        }
        Expr e = modChecksComplete(mge.precondition,callee_tprecondition,ev,
            loccall,calleeLoc,callerLoc,genExpr);
        if (genExpr && e != null) eev.addElement(e);
      }  // end of iterating over spec cases
    } finally {
      pFreshResult = false;
    }
    return !genExpr ? null : eev.size() == 0 ? null : GC.and(eev);
  }
  

  /** This adds checks for whether the given array with the given
   * index may be assigned to.
   * 
   * @param array the array object whose element is being assigned
   * @param arrayIndex the index of the elemtn being assigned
   * @param callLoc the location of the assignment
   */  
  void modifiesCheckArray(Expr array, Expr arrayIndex, int callLoc) {
    if (!issueCautions) return;
    kindOfModCheck = "assignment";
    modifiesCheckArray(array,arrayIndex,callLoc,Location.NULL,null,false,null,null,null);
  }

  /** This adds checks for whether the given array with the given
   * index may be assigned to.
   * 
   * @param array the array object whose element is being assigned
   * @param arrayIndex the index of the elemtn being assigned
   * @param callLoc the location of the assignment
   * @param calleeLoc
   * @param callee_tprecondition The precondition under which the callee might
   *          assign this array element; null is equivalent to true
   * @param genExpr true if an Expr containing the formula to be checked is
   *          to be returned, false if checks are to be inserted in the code
   * @param mg  the set of specifications against which the assignment of the
   *          field is to be tested
   * @param thisexpr the expression to be used for 'this' in translating
   *          input expressions, or null if no translation is to be done
   * @param args the set of variable mappings to be used in translation
   * @return the expression that must be true to allow the assignment, if 
   *    genExpr is true; if genExpr is false, null is returned
   */
  private Expr modifiesCheckArray(Expr array, Expr arrayIndex, 
      int callLoc, int calleeLoc,
      Expr callee_tprecondition,
      boolean genExpr, ModifiesGroupPragmaVec mg,
      Expr thisexpr, Hashtable args) {
    
    if (mg == null) mg = GetSpec.getCombinedMethodDecl(rdCurrent).modifies;
    ExprVec eev = ExprVec.make(mg.size());
     
    // FIXME - I don't think this handles this and super that are not the
    // prefix.

    // We need to create a translated predicate expression that is true
    // if the lhs is allowed to be modified and false if it is not
    // allowed to be modified.  Each specification case must be
    // satisfied.  Within a specification case there will be an OR across 
    // each of the store-ref expressions within the ModifiesGroupPragma
    // Each store-ref expression will contribute
    //   - obviously false if it is not the same declaration or same type
    //   - obviously true in some cases, such as if the store-ref location
    //      is \everything
    //   - some expression which is the AND of the condition and that
    //  the lhs matches the modifies expression:
    //      fields must have the same object designator
    //      arrays must have the same array expression and same index

    for (int kk=0; kk<mg.size(); ++kk) {
      ModifiesGroupPragma mge = mg.elementAt(kk);
      int callerLoc = mge.clauseLoc;
      // The composite condition is the OR of everything in ev.
      // This is false if there is nothing in ev.
      // ev is set to null if a literal TRUE would be added
      ExprVec ev = ExprVec.make(10);
      // The assignment is ok if the array whose element is assigned
      // is fresh since the prestate
      if (!genExpr) addAllocExpression(ev,array); 
      ModifiesIterator caller_iterator = new ModifiesIterator(
					     rdCurrent.parent,mge.items,true);
      while (ev != null && caller_iterator.hasNext()) {
        Object ex = caller_iterator.next();
        if (ex instanceof FieldAccess) {
          // skip - no match
        } else if (ex instanceof FieldDecl) {
          // skip - no match
        } else if (ex instanceof ArrayRefExpr) {
          if (array != null) {  // FIXME - why would array be null?
            Expr ao = ((ArrayRefExpr)ex).array;
            Expr ai = ((ArrayRefExpr)ex).index;
            ao = modTranslate(ao,!genExpr,thisexpr,args);
            ai = modTranslate(ai,!genExpr,thisexpr,args); 
            ao = GC.nary(TagConstants.REFEQ,array,ao);
            ai = GC.nary(TagConstants.INTEGRALEQ,arrayIndex,ai);
            ao = GC.and(ao,ai);
            ev.addElement(ao);
          }
        } else if (ex instanceof NothingExpr) {
          // skip - no match
        } else if (ex instanceof EverythingExpr) {
          ev = null;
        } else if (ex instanceof ArrayRangeRefExpr) {
          if (array != null) { // FIXME - why would array be null?
            ArrayRangeRefExpr a = (ArrayRangeRefExpr)ex;
            Expr ao = modTranslate(a.array,!genExpr,thisexpr,args);
            Expr al = a.lowIndex == null ? null :
              modTranslate(a.lowIndex,!genExpr,thisexpr,args); 
            Expr ah = a.highIndex == null ? null :
              modTranslate(a.highIndex,!genExpr,thisexpr,args); 
            ao = GC.nary(TagConstants.REFEQ,array,ao);
            al = al == null ? null :
              GC.nary(TagConstants.INTEGRALLE,al,arrayIndex);
            ah = ah == null ? null :
              GC.nary(TagConstants.INTEGRALLE,arrayIndex,ah);
            al = al == null ? ah : ah == null ? al :
              GC.and(al,ah);
            ao = al == null ? ao : GC.and(ao,al);
            ev.addElement(ao);
          }
        } else if (ex instanceof WildRefExpr) {
          // skip - an array reference does not match against a <obj>.*
        } else {
          ErrorSet.caution("INTERNAL ERROR: Unhandled store-ref type in Frame.modifiesCheckArray: " + ex.getClass());
        }
      }
      if (ev != null) {
        Expr e = modChecksComplete(mge.precondition,
            callee_tprecondition,ev,callLoc,
            calleeLoc==Location.NULL ? callerLoc : calleeLoc,
                calleeLoc==Location.NULL ? Location.NULL: callerLoc,genExpr);
        if (genExpr && ev != null) eev.addElement(e);
      }
    }
    return !genExpr ? null : eev.size() == 0 ? null : GC.and(eev);
  }
  

  /** Adds an expression into ev stating that e is allocated now but was
   * not allocated in the pre-state.  No expression is added to ev if it
   * is definitely false that e is fresh - such as if e is this or is null
   * (meaning the field is static).  Otherwise if pFreshResult is true,
   * a literal True expression is added.  Otherwise some non-trivial 
   * expression is added.
   * <p>
   * Also uses pFreshResult, which must be true iff e is known to be fresh
   * since the prestate.
   * @param ev A vector to accumulate assertions
   * @param e  An expression to be tested for freshness
   */
  private void addAllocExpression(ExprVec ev, Expr e) {
    if (e == null) return;
    if (e instanceof ThisExpr) return;
    if ((e instanceof VariableAccess) && 
        (((VariableAccess)e).id == thisId)) return;
    if (pFreshResult) {
      ev.addElement(GC.truelit);
      return;
    }
    ev.addElement (
        GC.and(
            GC.nary(TagConstants.ISALLOCATED,e,
                TrAnExpr.trSpecExpr(GC.allocvar)),
                GC.not(GC.nary(TagConstants.ISALLOCATED,e,
                    TrAnExpr.trSpecExpr(GC.allocvar,premap,premap)))
        )
    );
  }
   
  /**
   * Generates the actual check with the condition
   *   'if (precondition && tprecond2) then (disjunction of ev)'
   * 
   * @param precondition A precondition, not yet translated
   * @param tprecond2  A second, already translated, precondition
   *            (possibly null, meaning true)
   * @param ev  Disjunction of conditions to be proven; an empty list means false 
   * @param callLoc  Location of the assignment statement or method call
   * @param aloc  First associated location, possibly null
   * @param aloc2  Second associated location, possibly null
   * @param genExpr if true, returns an Expr that must be true; 
   *          if false, creates a check for that expression and
   *          returns null
   * @return The expr to be proved true, if genExpr is true; if
   *          genExpr is false, returns null
   */
  private Expr modChecksComplete(
      /*@ non_null */ Expr precondition, 
      Expr tprecond2, 
      ExprVec ev, 
      int callLoc, int aloc, int aloc2, boolean genExpr) {
    if (ev == null) {
      // always true, so we don't need to prove anything
      return genExpr ? GC.truelit : null;
    }

    boolean genCheck = true;
    
    // Check to see if the modifies check is disabled in general or
    // for the callLoc or aloc lines - if so just exit
    if (!genExpr && NoWarn.getChkStatus(TagConstants.CHKMODIFIES,callLoc,aloc==Location.NULL?callLoc:aloc)
        != TagConstants.CHK_AS_ASSERT) {
      TrAnExpr.closeForClause();
      genCheck = false;
    }
    // If aloc2 is not null, check to see if the warning is disabled for that
    // line - if so just exit
    if (aloc2 != Location.NULL && !genExpr) {
      if (NoWarn.getChkStatus(TagConstants.CHKMODIFIES,callLoc,aloc2)
          != TagConstants.CHK_AS_ASSERT) {
        TrAnExpr.closeForClause();
        genCheck = false;
      }
    }
    if (!genExpr && !genCheck) return null;
    
    Expr tprecondition = modTranslate(precondition,true,null,null);  // FIXME!!!
    if (tprecond2 != null  && !isTrueLiteral(tprecond2)) {
      tprecondition = GC.and(tprecondition, tprecond2);
    }

    Expr expr = 
        ev.size() != 0 ? GC.implies(tprecondition,GC.or(ev)) :
        !isTrueLiteral(tprecondition) ? GC.not(tprecondition) :
                          GC.falselit;
    if (genExpr) {
      // skip generating checks
    } else if (expr == GC.falselit) {
      // We need to prove (false), which we know without
      // benefit of the prover is false - so we immediately issue
      // an error
      if (aloc == TagConstants.NULL) {
        ErrorSet.error(callLoc, 
          "There is no assignable clause allowing this " +
          kindOfModCheck);
      } else {
        ErrorSet.error(callLoc, 
            "There is no assignable clause allowing this "
            + kindOfModCheck,aloc);
      }
      if (aloc2 != Location.NULL) ErrorSet.assocLoc(aloc2);
    } else if (aloc == Location.NULL) {
      //System.out.println("Generating a modifies check " + ev.size());    
      translator.addNewAssumptionsNow();
      translator.addCheck(callLoc,TagConstants.CHKMODIFIES, 
          expr);
    } else {
      //System.out.println("Generating a modifies check " + ev.size());    
      translator.addNewAssumptionsNow();
      translator.addCheck(callLoc,TagConstants.CHKMODIFIES, 
          expr,
          aloc,aloc2);
      // FIXME - could also include a list of locations from the caller modifies group
    }
    TrAnExpr.closeForClause();
    return genExpr ? expr : null;
  }
  
  /** Translates the Expr e into guarded commands:<BR>
   * if old is true, uses the premap to map variables
   * if old is false, uses args as the variable mapping
   * 
   * @param e  the Expr to translate
   * @param old if true, translates in the context of the pre-state (using the
   *    mapping in the class field 'premap'
   * @param thisexpr the Expr to use for 'this'
   * @param args the mapping to use if old is false (if args is not null)
   * @return the translated expression
   */
  private Expr modTranslate(
      /*@ non_null */Expr e, 
      boolean old, 
      Expr thisexpr, 
      Hashtable args) {
    TrAnExpr.initForClause(true);
    if (old) return TrAnExpr.trSpecExpr(e,premap,premap,null);
    else if (args == null)  return TrAnExpr.trSpecExpr(e,thisexpr);
    else     return TrAnExpr.trSpecExpr(e,args,args,thisexpr);
  }
  
  /** A utility function that returns true if the argument 
   * expression is null or strictly equal to a boolean TRUE.
   * 
   * @param p The expression to be tested
   * @return true if the argument is null or a Boolean LiteralExpr
   *      with value true
   */
  private boolean isTrueLiteral(Expr p) {
    if (p == null) return true;
    if (!(p instanceof LiteralExpr)) return false;
    LiteralExpr lit = (LiteralExpr)p;
    if (lit.getTag() != TagConstants.BOOLEANLIT) return false;
    Object value = lit.value;
    return ((Boolean)value).booleanValue() ;
  }
  
  /** This class enables iterating over the set of store-ref
   * locations in a ModifiesGroupPragma.  It also has the ability
   * to include in the iteration the contents of datagroups that
   * are part of the set of store-ref locations.
   * 
   * @author David Cok
   *
   */
  static class ModifiesIterator {

    /** The TypeDecl whose view of any datagroups is to be used.*/
    final private TypeDecl td;
    
    /** The set of store-ref locations over which to iterate. */
    final private CondExprModifierPragmaVec mp;
    
    /** Fields that have yet to be iterated over.  This
     * is a list of FieldAccess objects. */
    final private List others = new LinkedList();
    
    /** The datagroups that have already been expanded */
    final private List done = new LinkedList();
    
    /** If true, then datagroups are expanded and their contents
     * made part of the iteration.
     */
    final private boolean expandDatagroups;
    
    /** If true, then field wild card store refs (obj.* and
     * Type.*) are expanded and their contents made part of the
     * iteration.
     */
    final private boolean expandWild;
    
    /** An array index into mp */
    private int i = 0;
    
    /** Creates an iterator over the store-ref locations in
     * the CondExprModifierPragmaVec.  The expandWild parameter
     * is set false.
     * @param mp The set of store-ref locations over which to
     *      iterate
     * @param expandDatagroups if true, then datagroups are 
     *      expanded and their contents (recursively) become
     *      steps in the iteration
     */
    public ModifiesIterator(TypeDecl td,
                CondExprModifierPragmaVec mp, boolean expandDatagroups) {
      this(td,mp,expandDatagroups,false);
    }

    
    /** Creates an iterator over the store-ref locations in
     * the CondExprModifierPragmaVec.
     * @param mp The set of store-ref locations over which to
     *      iterate
     * @param expandDatagroups if true, then datagroups are 
     *      expanded and their contents (recursively) become
     *      steps in the iteration
     * @param expandWild if true, then store-ref expressions
     *      of the form  obj.* are expanded into their
     *      individual fields
     */
    public ModifiesIterator(TypeDecl td, CondExprModifierPragmaVec mp, 
        boolean expandDatagroups, boolean expandWild) {
      this.td = td;
      this.mp = mp;
      this.expandDatagroups = expandDatagroups;
      this.expandWild = expandWild;
      i = 0;
    }
    
    /** Resets the iteration back to the beginning */
    public void reset() {
      i = 0;
      others.clear();
      done.clear();
    }
    
    /** Returns true if there is more to the iteration 
     * @return true if there is more to the iteration
     */
    public boolean hasNext() {
      return i < mp.size() || !others.isEmpty();
    }
    
    /** Returns the next element of the iteration; only valid
     * if hasNext returns true, otherwise throws an exception
     * @return the next element of the iteration
     * @throws NoSuchElementException if there are no more elements
     *      in the iteration
     */
    public Object next() throws NoSuchElementException {
      Object ex;
      if (!others.isEmpty()) {
        ex = others.remove(0);
      } else if (i >= mp.size()) {
        throw new NoSuchElementException();
      } else {
        ex = mp.elementAt(i).expr;
        ++i;
        done.clear();
      }
      if (ex instanceof FieldAccess) {
        if (expandDatagroups) add(((FieldAccess)ex).od,((FieldAccess)ex).decl);
      } else if (expandWild && (ex instanceof WildRefExpr)) {
        //System.out.println("EXPANDING " + Location.toString(((WildRefExpr)ex).getStartLoc()));
        ObjectDesignator od = ((WildRefExpr)ex).od;
        addFields(od);
      }
      
      return ex;
    }
    
    /** The maximum number of times to unroll a maps reference. */
    private int limit = Main.options().mapsUnrollCount;
    // FIXME - should find a better way of testing than using a limited unrolling
    
    /** Adds all the fields of the od (whether it is a type
     * or an object) into the 'others' list as FieldAccess
     * items.
     * @param od
     */
    private void addFields(ObjectDesignator od) {
      Type type;
      boolean stat;
      if (od instanceof TypeObjectDesignator) {
        type = ((TypeObjectDesignator)od).type;
        stat = true;
      } else if (od instanceof ExprObjectDesignator) {
        type = TypeCheck.inst.getType(((ExprObjectDesignator)od).expr);
        stat = false;
      } else {
        ErrorSet.caution("INTERNAL ERROR: Unhandled ObjectDesignator (od) in ModifiesIterator.addFields: " + od.getClass());
        return;
      }
      if (type instanceof javafe.tc.TypeSig) {
        TypeSig ts = (TypeSig)type;
        EnvForTypeSig env;
        if (stat)
          env = (EnvForTypeSig)FlowInsensitiveChecks.staticenvDecoration.get(ts.getTypeDecl());
        else
          env = (EnvForTypeSig)FlowInsensitiveChecks.envDecoration.get(ts.getTypeDecl());
        if (env == null) env = ((TypeSig)type).getEnv(stat);
        // The true in the following means all fields are gotten,
        // whether or not they are visible or inherited.
        // It does also return ghost and model fields.
        javafe.ast.FieldDeclVec fds = env.getFields(true);
        for (int i=0; i<fds.size(); ++i) {
          FieldDecl fd = fds.elementAt(i);
          if (stat != Modifiers.isStatic(fd.modifiers)) continue;
          FieldAccess fa = FieldAccess.make(od,fd.id,Location.NULL);
          fa.decl = fd;
          fa = (FieldAccess)javafe.tc.FlowInsensitiveChecks.setType(fa,fd.type);
          others.add(fa);
          //System.out.println("ADDING " + fd.id + " " + fd.type);
        }
      } else {
        ErrorSet.caution("INTERNAL ERROR: Unhandled Type in ModifiesIterator.addFields: " + type.getClass());
      }
    }
    
    /** Adds the contents of the datagroup d (of object od, which
     * may not be null) to the 'others' list.
     * @param od Object reference
     * @param d  Declaration of the datagroup
     */
    //@ requires od != null && d != null;
    private void add(ObjectDesignator od, FieldDecl d) {
      if (count(d) >= limit) return;
      done.add(d);
      if (od == null) {
        System.out.println("UNSUPPORTED OPTION IN FRAME.ModifiesIterator");
        others.addAll(Datagroups.get(td,d));
      } else if (od instanceof TypeObjectDesignator) {
        TypeSig ts = (TypeSig)((TypeObjectDesignator)od).type;
        TypeDecl tdd = ts.getTypeDecl();
        if (TypeSig.getSig(td).isSubtypeOf(ts)) tdd = td;
        others.addAll(Datagroups.get(tdd,d));
      } else if (od instanceof ExprObjectDesignator) {
        Expr e = ((ExprObjectDesignator)od).expr;
        Type t = javafe.tc.FlowInsensitiveChecks.getType(e);
        
        TypeDecl tdd;
        if (t instanceof TypeSig) {
			tdd = ((TypeSig) t).getTypeDecl();
        
			if (TypeSig.getSig(td).isSubtypeOf((TypeSig) t))
				tdd = td;
			Iterator i = Datagroups.get(tdd, d).iterator();
			// FIXME - need to determine what the permissible content
			// of Datagroups.get() is
			Hashtable h = new Hashtable();
			h.put(Substitute.thisexpr, e);
			while (i.hasNext()) {
				Object o = i.next();
				if (o instanceof FieldDecl) {
					ErrorSet
							.caution("INTERNAL ERROR: Unhandled FieldDecl in ModifiesIterator.add: "
									+ o);
				} else if (o instanceof Expr) {
					Expr ee = (Expr) o;
					ee = Substitute.doSubst(h, ee);
					others.add(ee);
				} else {
					ErrorSet
							.caution("INTERNAL ERROR: Unhandled case in ModifiesIterator.add: "
									+ o.getClass());
				}
			}
        }
			else {
				ErrorSet
						.caution("INTERNAL ERROR: Unhandled type in ModifiesIterator.add: "
								+ t.getClass());
			}
      } else if (od instanceof SuperObjectDesignator) {
        TypeSig ts = (TypeSig)((SuperObjectDesignator)od).type;
        TypeDecl tdd = ts.getTypeDecl();
        if (TypeSig.getSig(td).isSubtypeOf(ts)) tdd = td;
        others.addAll(Datagroups.get(tdd,d));
      }
    }
    
    /** Returns the number of times the argument is in the 'done'
     * list
     * @param d Object to be checked
     * @return The number of times the argument is in the 'done' list
     */
    private int count(FieldDecl d) {
      int k = 0;
      Iterator i = done.iterator();
      while (i.hasNext()) {
        if (i.next() == d) ++k;
      }
      return k;
    }
    
  }
  

}
