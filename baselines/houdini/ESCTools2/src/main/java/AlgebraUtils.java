package escjava.dfa.daganalysis;

/*
 * Created on Mar 16, 2006
 */

import java.util.HashMap;

//import javafe.ast.*;
import javafe.ast.ASTNode;
import javafe.ast.Expr;
import javafe.ast.ExprVec;
import javafe.ast.BinaryExpr;
import javafe.ast.GenericVarDecl;
import javafe.ast.LiteralExpr;
import javafe.ast.VariableAccess;
import javafe.util.Assert;

//import escjava.ast.*;
import javafe.ast.GenericVarDeclVec;
import escjava.ast.NaryExpr;
import escjava.ast.QuantifiedExpr;
import escjava.ast.LabelExpr;
import escjava.ast.TypeExpr;

import escjava.prover.Atom;

import escjava.translate.GC;
import escjava.translate.UniqName;

//Debugging purposes
import escjava.translate.VcToString;

import escjava.ast.TagConstants;

/**
 * @author Mikolas Janota
 * 
 * This class encapsulates algebra-like operations on expressions and related data-structures.
 */
public class AlgebraUtils {

    /**
     * Computes the set difference for the given var declaration vectors.
     * 
     * @param a from this vector we subtract
     * @param b this vector will be subtracted
     * @result a new vector that will contain the difference (alway non-null),
     *         i.e., it will contain those elements from <code>a</code> that
     *         are not in <code>b</code>
     */
    //@ ensures \fresh(\result);
    /*@ pure */public static /*@non_null */GenericVarDeclVec difference(/*@ non_null */GenericVarDeclVec a, /*@ non_null */
            GenericVarDeclVec b) {
        GenericVarDeclVec retv = GenericVarDeclVec.make();
        for (int i = 0; i < a.size(); i++) {
            GenericVarDecl vd = a.elementAt(i);

            if (!b.contains(vd)) {
                // add the vd in the result only if it's not in b
                retv.addElement(vd);
            }
        }
        return retv;
    }
    
    /*@ pure */public static boolean contains(/*@ non_null*/GenericVarDeclVec varVec, /*@ non_null*/GenericVarDecl var) {
        for (int i = 0; i < varVec.size(); i++) {
            GenericVarDecl vi = varVec.elementAt(i);
            if (var == vi) return true;
        //    if (vi.id.equals(var.id) && vi.locId == var.locId) 
          //      return true;
        }
        return false;
    }

    /*@ pure*/public static boolean contains(/*@ non_null*/ExprVec ev, /*@ non_null*/Expr e) {
        for (int i = 0; i < ev.size(); i++) {
            if (isSame(ev.elementAt(i), e))
                return true;
        }
        return false;
    }
    
    //@ ensures \fresh(\result);
    /*@ pure*/public static /*@ non_null*/ExprVec difference(
            /*@ non_null*/ExprVec a, 
            /*@ non_null*/ExprVec b) {
        ExprVec diff = ExprVec.make();
        for (int i = 0; i < a.size(); i++) {            
            Expr e = a.elementAt(i);

            if (!contains(b, e)) {
                // add the e in the result only if it's not in b
                diff.addElement(e);
            }
        }
        return diff;
    }

    
    /*@ pure*/public static boolean subset(/*@ non_null*/GenericVarDeclVec vec1, /*@ non_null*/GenericVarDeclVec vec2) {
        for (int i = 0; i < vec1.size(); i++) {
            GenericVarDecl vd1 = vec1.elementAt(i);

            if (!vec2.contains(vd1)) {
                return false;
            }
        }
        return true;
    }

    /*@ pure*/public static boolean setEquals(/*@ non_null*/GenericVarDeclVec vec1, /*@ non_null*/GenericVarDeclVec vec2) {
        return subset(vec1, vec2) && subset(vec2, vec1);
    }

    public static boolean intersect(GenericVarDeclVec vec1, GenericVarDeclVec vec2) {
        for (int i = 0; i < vec1.size(); i++) {
            GenericVarDecl decl1 = vec1.elementAt(i);
            if (vec2.contains(decl1))
                return true;
        }
        return false;
    }

    static void andAdd(ExprVec ev, Expr e) {
        if (e.getTag() == TagConstants.BOOLAND) {
            NaryExpr nex = (NaryExpr) e;
            ExprVec toAdd = nex.exprs;
            setAppend(ev, toAdd);
        } else {
            setAdd(e, ev);
        }

    }

    static void orAdd(ExprVec ev, Expr e) {
        if (e.getTag() == TagConstants.BOOLOR) {
            NaryExpr nex = (NaryExpr) e;
            ExprVec toAdd = nex.exprs;
            setAppend(ev, toAdd);
        } else {
            setAdd(e, ev);
        }
    }

    private static ExprVec isEQ(/*@ non_null*/Expr e) {
        if (e.getTag() == TagConstants.ANYEQ) {
            ExprVec terms = ((NaryExpr) e).exprs;
            Assert.notFalse(terms.size() == 2, "I guess I don't understand this op.");
            
            return terms;
        }
        
        return null;
    }
    
    //@ ensures \result != null;
    //@ ensures \result.size() == target.size();
    static ExprVec subst(/*@ non_null*/GenericVarDecl vd, /*@ non_null*/Expr val, /*@ non_null*/ExprVec target) {
        ExprVec retv = ExprVec.make(target.size());
        for (int i = 0; i < target.size(); i++) {
            Expr te = target.elementAt(i);
            te = GC.subst(vd, val, te);
            retv.addElement(te);
        }
        
        return retv;
    }

    //@ ensures e != null ==> \result != null;
    private static Expr removeLetExpression(Expr e) {
        if (!(e instanceof QuantifiedExpr))
            return e;

        QuantifiedExpr qe = (QuantifiedExpr) e;

        // universal quantifiers only
        if (qe.quantifier != TagConstants.FORALL)
            return e;

        
        Expr inside = qe.expr;
        // continaing implications
        if (inside.getTag() != TagConstants.BOOLIMPLIES)
            return e;

        
        ExprVec exprs = ((NaryExpr) inside).exprs;
        Expr left = exprs.elementAt(0);
        Expr right = exprs.elementAt(1);
        ExprVec les = flattenAnd(left); // brake left side into conjuncts
        for (int i = 0; i < les.size();) {
            Expr le = les.elementAt(i);

            ExprVec eqsides = isEQ(le);
            if (eqsides != null) {
                GenericVarDecl leftE = null;
                Expr rightE = null;
                GenericVarDecl v1 = eqsides.elementAt(1) instanceof VariableAccess ? ((VariableAccess) eqsides.elementAt(1)).decl : null;
                GenericVarDecl v0 = eqsides.elementAt(0) instanceof VariableAccess ? ((VariableAccess) eqsides.elementAt(0)).decl : null;

                // look for a side of the equality that is a variable quantified over in this expression
                if (v1 != null && qe.vars.contains(v1)) {
                    leftE = v1;
                    rightE = eqsides.elementAt(0);
                } else if (v0 != null && qe.vars.contains(v0)) {
                    leftE = v0;
                    rightE = eqsides.elementAt(1);
                }

                if (leftE != null) {
                    // at least one side of the equality is quantified over
                    if (!usesVar(leftE, rightE)) {// check if leftE is not continained in rightE                        
                        // remove the element the equality from the left part of the implication
                        les.removeElementAt(i);

                        // apply substituion on the left part of the implication
                        les = subst(leftE, rightE, les);

                        // apply substituion on the right part of the  implication
                        right = GC.subst(leftE, rightE, right);

                        continue;
                    }
                }

            }

            i++; // this is skipped when an element is removed from [les], due to the [continue]
        }

        Expr retv;

        if (les.size() == 0) {
            // all the elemnts on the lefthand side were removed
            retv = right;
        } else {
            // recreate the implication
            retv = GC.implies(andSimple(les), right);
        }
        
        // recreate the quantified expression
        QuantifiedExpr qeRetv= recreateQuantifiedExpr(qe, retv);
        return qeRetv;
    }
    
    
    /**
     * And two expressions in such a way that the n-ary and is exploited.
     * 
     * @param e1 an <code>Expr</code> value
     * @param e2 an <code>Expr</code> value
     * @return expression representing the and value of the given parameters
     */
    public static Expr and(Expr e1, Expr e2) {        
        ExprVec retvExprs = ExprVec.make();
        andAdd(retvExprs, e1);
        andAdd(retvExprs, e2);
        Expr retv = GC.and(retvExprs);
        retv = normalizeAnd(retv);
        return retv;
    }
    
    public static /*@ non_null */Expr orSimple(/*@ non_null */ExprVec ev) {
        switch (ev.size()) {
        case 0: return GC.falselit;
        case 1: return ev.elementAt(0);
        default: return GC.or(ev);        
        }
    }
    
    public static /*@ non_null */Expr andSimple(/*@ non_null */ExprVec ev) {
        switch (ev.size()) {
        case 0: return GC.truelit;
        case 1: return ev.elementAt(0);
        default: return GC.and(ev);
        }         
    }

    public static /*@ non_null */QuantifiedExpr recreateQuantifiedExpr(/*@ non_null */QuantifiedExpr qexpr,
            /*@ non_null */Expr newBoundExpr) {
        return QuantifiedExpr.make(qexpr.sloc, qexpr.eloc, qexpr.quantifier,
                qexpr.vars, qexpr.rangeExpr, newBoundExpr, 
                qexpr.nopats, qexpr.pats);
    }
    
    private static/*@ non_null */QuantifiedExpr recreateQuantifiedExprLazy(
            /*@ non_null */QuantifiedExpr qexpr,
            /*@ non_null */Expr newBoundExpr) {
        if (qexpr.expr == newBoundExpr) {
            return qexpr;
        }
        else {
            return QuantifiedExpr.make(qexpr.sloc, qexpr.eloc,
                    qexpr.quantifier, qexpr.vars, qexpr.rangeExpr,
                    newBoundExpr, qexpr.nopats, qexpr.pats);
        }
    }


    public static NaryExpr recreateNary(NaryExpr old, ExprVec newChildren) {
        NaryExpr retv = NaryExpr.make(old.sloc, old.eloc, old.op, old.methodName, newChildren);
        return retv;
    }
    
    public static /*@ non_null */LabelExpr recreateLabelExpr(/*@ non_null */LabelExpr le, /*@ non_null */Expr newInside) {
        return LabelExpr.make(le.sloc, le.eloc, le.positive, le.label, newInside);      
    }

    
    private static /*@ non_null */LabelExpr recreateLabelExprLazy(/*@ non_null */LabelExpr le, /*@ non_null */Expr newInside) {
        if (le.expr == newInside) {
            return le;
        } else {
            return LabelExpr.make(le.sloc, le.eloc, le.positive, le.label, newInside);
        }
    }


    /*@ pure */private static boolean isComplementaryNary(int tag1, int tag2) {        
         final int[][] complementary = 
           { {TagConstants.ANYEQ, TagConstants.ANYNE },
             { TagConstants.INTEGRALEQ, TagConstants.INTEGRALNE}, 
             { TagConstants.REFEQ, TagConstants.REFNE},
             { TagConstants.FLOATINGEQ, TagConstants.FLOATINGNE }
             };
         for (int i = 0; i < complementary.length; i++) {
             if (complementary[i][0] == tag1)
                 return complementary[i][1] == tag2;
             if (complementary[i][0] == tag2)
                 return complementary[i][1] == tag1;             
         }
         
         return false;         
    }
    
    public static boolean isComplementary(Expr e1, Expr e2) {        
        if (isComplementaryNary(e1.getTag(), e2.getTag())) {
            ExprVec ev1 = ((NaryExpr) e1).exprs;
            ExprVec ev2 = ((NaryExpr) e2).exprs;
            return  isSame(ev1, ev2);
            }
            
        if (e1.getTag() == TagConstants.BOOLNOT) {
            NaryExpr nex = (NaryExpr) e1;
            e1 = nex.exprs.elementAt(0);
        } else {
            if (e2.getTag() == TagConstants.BOOLNOT) {
                NaryExpr nex = (NaryExpr) e2;
                e2 = nex.exprs.elementAt(0);
            } else {
                return false;
            }
        }

        return isSame(e1, e2);

    }

    public static boolean containsComplementary(ExprVec evec) {
        for (int i = 0; i < evec.size() - 1; i++) {
            for (int j = i + 1; j < evec.size(); j++) {
                Expr e1 = evec.elementAt(i);
                Expr e2 = evec.elementAt(j);
                if (isComplementary(e1, e2)) {
                    return true;
                }
            }
        }
        return false;
    }

  
    /**
     * Strips off labels from the given expression.
     * 
     * @param pred the expression from which the labels will be stripped off
     * @result expression equivalent to the given one without the outter labels
     */
    /*@ pure */public static Expr stripOffLabels(/*@ non_null*/Expr pred) {
        Expr retv = pred;
        while (retv.getTag() == TagConstants.LABELEXPR) {
            LabelExpr le = (LabelExpr) retv;
            retv = (le.expr);
        }

        return retv;
    }

    /**
     * Test whether the specified expression is a boolean literal with the specified
     * value.
     * 
     * @param expr   tested expression
     * @param value  a <code>boolean</code> value representing the desired value
     *            of the expression
     * @return a <code>boolean</code> value indicating that the given
     *         expression is a boolean literal with the given value
     */
    /*@ pure */public static boolean isBooleanLiteral(/*@ non_null*/Expr expr, boolean value) {
      //  expr = stripOffLabels(expr);
        if (expr instanceof LiteralExpr) {
            LiteralExpr le = (LiteralExpr) expr;
            Object storedValue = le.value;
            if (storedValue instanceof Boolean) {
                Boolean storedBoolean = (Boolean) storedValue;
                return storedBoolean.booleanValue() == value;
            } else {
                return false;
            }
        } else {
            return false;
        }
    }

    /**
     * Determines whether a given expression is the true literal.
     * @param expr the inspected expression
     * @return returns true if and only if the given expression is a literal a
     *         expression and it is true
     */
    /*@ pure */public static boolean isTrueLit(/*@ non_null*/Expr expr) {
        return isBooleanLiteral(expr, true);
    }

    /**
     * Determines whether a given expression is the false literal. 
     * @param expr the inspected expression
     * @return returns true if and only if the given expression is a literal a
     *         expression and it is false
     */
    //@ requires expr != null;
    /*@ pure */public static boolean isFalseLit(Expr expr) {
        return isBooleanLiteral(expr, false);
    }

    /**
     * Computes whether an expression contains a reference to a given variable.
     * @param varDecl the tested variable
     * @param e the tested expression
     * @return returns true if and only of <code>e</code> references
     *         <code>varDecl</code>
     */
    public static boolean usesVar(GenericVarDecl varDecl, Expr e) {
        //GenericVarDeclVec eVars = getVars(e);
        //return contains(eVars, varDecl);
        return usesVar(e, varDecl);
    }
    
    
    private static boolean usesVar(ASTNode n, GenericVarDecl varDecl) {
        if (n instanceof VariableAccess) {
            VariableAccess va = (VariableAccess) n;
            return varDecl == va.decl;
        } else {
            if (n instanceof QuantifiedExpr) {
                QuantifiedExpr q = (QuantifiedExpr) n;
                GenericVarDeclVec boundedVars = q.vars;
               
                if (contains(boundedVars, varDecl))
                   return false;
                
                return usesVar(q.expr, varDecl);                
            } else {
                int childCount = n.childCount();
                for (int i = 0; i < childCount; i++) {
                    Object child = n.childAt(i);
                    if (child instanceof ASTNode) {
                        ASTNode childAST = (ASTNode) child;
                        if (usesVar(childAST, varDecl))
                            return true;                        
                    }
                }
            }

        }    
        
        return false;
    }

    static void remove(GenericVarDeclVec a, GenericVarDeclVec b) {
        for (int i = 0; i < b.size(); i++) {
            GenericVarDecl gvd = b.elementAt(i);
            a.removeElement(gvd);
        }
    }

    /**
     * Tests whether a given expression is using any of the given variables.
     * 
     * @param varDecls
     *            a <code>GenericVarDeclVec</code> value
     * @param e
     *            an <code>Expr</code> value
     * @return a <code>boolean</code> value
     */
    public static boolean usesAnyVar(GenericVarDeclVec varDecls, ASTNode n) {
        if (n instanceof VariableAccess) {
            VariableAccess va = (VariableAccess) n;
            return contains(varDecls, va.decl);
        } else {
            if (n instanceof QuantifiedExpr) {
                QuantifiedExpr q = (QuantifiedExpr) n;
                GenericVarDeclVec boundedVars = q.vars;
               
                if (intersect(boundedVars, varDecls))
                   return false;
                GenericVarDeclVec fvs = difference(varDecls, boundedVars);
                return usesAnyVar(fvs, q.expr);                
            } else {
                int childCount = n.childCount();
                for (int i = 0; i < childCount; i++) {
                    Object child = n.childAt(i);
                    if (child instanceof ASTNode) {
                        ASTNode childAST = (ASTNode) child;
                        if (usesAnyVar(varDecls, childAST))
                            return true;                        
                    }
                }
            }

        }    
        
        return false;
    }

    /**
     * Add to the given vector  variable declarations for variables
     * referenced in the given expression.
     */
    static void getVars(ASTNode n, GenericVarDeclVec vec) {
        if (n instanceof VariableAccess) {
            VariableAccess va = (VariableAccess) n;
            GenericVarDecl varDecl = va.decl;

            if (!vec.contains(varDecl))
                vec.addElement(varDecl);
        } else {
            if (n instanceof QuantifiedExpr) {
                QuantifiedExpr q = (QuantifiedExpr) n;
                GenericVarDeclVec boundedVars = q.vars;
                GenericVarDeclVec insideVars = getVars(q.expr);
                setAppend(vec, difference(insideVars, boundedVars));
            } else {
                int childCount = n.childCount();
                for (int i = 0; i < childCount; i++) {
                    Object child = n.childAt(i);
                    if (child instanceof ASTNode) {
                        ASTNode childAST = (ASTNode) child;
                        getVars(childAST, vec);
                    }
                }
            }

        }
    }

    public static /*@ non_null*/GenericVarDeclVec getVars(ASTNode n) {
        GenericVarDeclVec vec = GenericVarDeclVec.make();
        getVars(n, vec);
        return vec;
    }
    
    public static boolean isGround(ASTNode n) {
        return getVars(n).size() == 0;
    }

    public static boolean shareVariables(Expr e1, Expr e2) {
        GenericVarDeclVec vec1 = getVars(e1);
        GenericVarDeclVec vec2 = getVars(e2);
        boolean doIntersect = intersect(vec1, vec2);

        /*
         * reportln("Intersect test: " + doIntersect);
         * PredicateAbstraction.printlnExpr(e1);
         * PredicateAbstraction.printlnExpr(e2);
         */

        return doIntersect;
    }

    public static boolean shareVariables(ExprVec es, Expr e) {
        /*
         * reportln("== test for variable sharing ");
         * PredicateAbstraction.printlnExprVec(es); reportln(" and ");
         * PredicateAbstraction.printlnExpr(e);
         */

        for (int i = 0; i < es.size(); i++) {
            Expr e1 = es.elementAt(i);

            if (shareVariables(e, e1))
                return true;
        }
        return false;
    }

    public static boolean predContain(Expr pred, ExprVec predVec) {
        for (int i = 0; i < predVec.size(); i++) {
            if (isSame(predVec.elementAt(i), pred)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Adds a given expression to a given expression vector but only when it's
     * not already contained.
     * 
     * @param pred the expression to be added
     * @param predVec   the vector in which to add
     */
    public static void setAdd(Expr pred, ExprVec predVec) {
        boolean alreadyThere = contains(predVec, pred);
        if (!alreadyThere) {
            predVec.addElement(pred);
        }
    }

    
    static void setAppend(ExprVec ev, ExprVec appendix) {
        for (int i = 0; i < appendix.size(); i++) {
            Expr e = appendix.elementAt(i);
            setAdd(e, ev);
        }
    }
    
   public static void setAdd(GenericVarDecl vd, GenericVarDeclVec vdVec) {
       boolean alreadyThere = false;
       for (int i = 0; i < vdVec.size(); i++) {
           if (vdVec.elementAt(i).id.equals(vd.id)) {
               alreadyThere = true;
               break;
           }
       }
       if (!alreadyThere) {
           vdVec.addElement(vd);
       }
   }
   
   public static void setAppend(GenericVarDeclVec vdVec, GenericVarDeclVec appendix) {
       for (int i = 0; i < appendix.size(); i++) {
           GenericVarDecl vd = appendix.elementAt(i);
           setAdd(vd, vdVec);
       }
   }
  

    public static ExprVec join(ExprVec ev1, ExprVec ev2) {
        ExprVec retv = ExprVec.make();
        setAppend(retv, ev1);
        setAppend(retv, ev2);
        return retv;
    }

    /**
     * Creates a copy of the given expression vector but duplicated expressions
     * are included just once. The method <code>isSame</code> is used as the
     * equality test.
     */
    public static ExprVec removeDuplicates(ExprVec evec) {
        ExprVec cleanPredicates = ExprVec.make(evec.size());

        for (int i = 0; i < evec.size(); i++) {
            Expr e = evec.elementAt(i);
            setAdd(e, cleanPredicates);
        }

        return cleanPredicates;
    }

    /**
     * From a set of expression returns a subset of such expressions that do not
     * refer to the specified set of (irelevant) variables.
     * 
     * @param oldExprs  the set to be filtered
     * @param nonRelevantVariables the set of (irelevant) variables
     * @return filtered expression set, if the given expression is
     *         <code>null</code> ther return value is null, otherwise it is
     *         non-null
     */
    private static ExprVec filterIrelevantExpressions(ExprVec oldExprs,
            GenericVarDeclVec nonRelevantVariables) {
        if (oldExprs == null) {
            return null;
        }

        ExprVec filteredExprs = ExprVec.make();

        for (int i = 0; i < oldExprs.size(); i++) {
            Expr oldExpr = oldExprs.elementAt(i);
            if (!AlgebraUtils.usesAnyVar(nonRelevantVariables, oldExpr)) {
                // the pattern doesn't refer to any of the irelevant variables
                filteredExprs.addElement(oldExpr); // it is added to the new
                                                    // set
            }

        }
        return filteredExprs;
    }

    /**
     * For a given quantified expression, the variables that are quantified over
     * are split into two sets according to whether they are relevant or
     * irrelevant to the inside of the quantified expression.
     * 
     * @param qe  the quantified expression to be tested
     * @param relevantVars  to this vector the relevant vars will be added
     * @param nonRelevantVars to this vector the irelevant vars will be added
     */
    //@ requires qe != null && relevantVars != null && nonRelevantVars != null;
    //@ ensures qe.vars.size() == relevantVars.size() + nonRelevantVars.size();
    public static void relevantVariablesSplit(QuantifiedExpr qe,
            GenericVarDeclVec relevantVars, GenericVarDeclVec nonRelevantVars) {
        Expr boundExpr = qe.expr;
        GenericVarDeclVec vars = qe.vars;

        for (int i = 0; i < vars.size(); i++) {
            GenericVarDecl varDecl = vars.elementAt(i);
            boolean isRelevant = AlgebraUtils.usesVar(varDecl, boundExpr);
            if (qe.rangeExpr != null) {
                if (AlgebraUtils.usesVar(varDecl, qe.rangeExpr))
                    isRelevant = true; // is this optimal?
            }

            GenericVarDeclVec addToVec;
            if (isRelevant) {
                addToVec = relevantVars;
            } else {
                addToVec = nonRelevantVars;
            }
            addToVec.addElement(varDecl);
        }
    }

    /**
     * Compares two expression vectors using <code>isSame</code> method to
     * compare the elements.
     */
    private static boolean isSame(ExprVec v1, ExprVec v2, HashMap varBinding) {
        if (v1.size() != v2.size())
            return false;

        for (int i = 0; i < v1.size(); i++) {
            boolean areSame = isSame(v1.elementAt(i), v2.elementAt(i), varBinding);
            if (!areSame) {
                return false;
            }
        }

        return true;
    }

    private static boolean isSameNullSafe(Expr p1, Expr p2, HashMap varBinding) {
        if ((p1 == null && p2 != null) || (p1 != null && p2 == null)) {
            return false;
        } else {
            // either both non-null or both null
            if (p1 != null && p2 != null && !isSame(p1, p2, varBinding))
                return false; // both non-null but different
        }

        return true;
    }
    
    public static boolean isSame(ExprVec ev1, ExprVec ev2) {
      return isSame(ev1, ev2, null);    
    }
    
    public static boolean isSame(Expr p1, Expr p2) {
      return isSame(p1, p2, null);
    }

    public static boolean isSame(Expr p1, Expr p2, HashMap varBinding) {
        //p1 = stripOffLabels(p1);
        //p2 = stripOffLabels(p2);

        boolean retv;
        if (p1.getTag() != p2.getTag()) {
            retv = false;
        } else {
            if ((p1 instanceof VariableAccess)
                    && (p2 instanceof VariableAccess)) {
                VariableAccess va1 = (VariableAccess) p1;
                VariableAccess va2 = (VariableAccess) p2;
                GenericVarDecl va1Binding = null;
                if (varBinding != null) {
                     va1Binding =  (GenericVarDecl) varBinding.get(va1.decl);
                }
                if (va1Binding != null) {// there's a binding for this var
                    retv = (va2.decl == va1Binding);
                    //retv = va2.decl.id.equals(va1Binding.id) && (va2.decl.locId == va1Binding.locId);;
                } else {
                    retv = (va1.decl == va2.decl);
                   //retv = (va1.id).equals(va2.id) && (va1.decl.locId == va2.decl.locId);
                }
            } else {
                if (p1 instanceof NaryExpr && p2 instanceof NaryExpr) {
                    NaryExpr n1 = (NaryExpr) p1;
                    NaryExpr n2 = (NaryExpr) p2;
                    retv = (n1.op == n2.op); 
                    retv &= ((n1.methodName == null && n2.methodName == null)) || n1.methodName.equals(n2.methodName);
                    retv &= isSame(n1.exprs, n2.exprs, varBinding);
                } else {
                    if (p1 instanceof TypeExpr && p2 instanceof TypeExpr) {
                        TypeExpr te1 = (TypeExpr) p1;
                        TypeExpr te2 = (TypeExpr) p2;
                        retv = (te1.eloc == te2.eloc)
                                && te1.type.equals(te2.type);
                    } else {
                        if (p1 instanceof QuantifiedExpr
                                && p2 instanceof QuantifiedExpr) {
                            QuantifiedExpr q1 = (QuantifiedExpr) p1;
                            QuantifiedExpr q2 = (QuantifiedExpr) p2;

                            HashMap qBinding;
                            if (varBinding != null) {
                                qBinding = (HashMap)varBinding.clone();    
                            } else {
                                qBinding = new HashMap();
                            }
                            
                            GenericVarDeclVec vars1 = q1.vars;
                            GenericVarDeclVec vars2 = q2.vars;
                            
                            if ((vars1.size() != vars2.size()) || (q1.quantifier != q2.quantifier)) {
                                retv = false;
                            } else {
//                                retv = true;
//                                for (int i = 0; i < q1.vars.size(); i++) {
//                                    if (vars1.elementAt(i) != vars2.elementAt(i)) {
//                                        retv = false;
//                                        break;
//                                    }                                        
//                                }
//                                retv =  retv && isSameNullSafe(q1.rangeExpr, q2.rangeExpr, varBinding) &&
//                                        isSame(q1.expr, q2.expr, varBinding);
                             
                                for (int i = 0; i < vars1.size(); i++) {
                                    qBinding.put(vars1.elementAt(i), vars2.elementAt(i));
                                }
                                retv = isSameNullSafe(q1.rangeExpr, q2.rangeExpr, qBinding)
                                       && isSame(q1.expr, q2.expr, qBinding);
                            }
                                
                            
                            
                        } else {
                            if (p1 instanceof LiteralExpr && p2 instanceof LiteralExpr) {
                                LiteralExpr le1 = (LiteralExpr) p1;
                                LiteralExpr le2 = (LiteralExpr) p2;
                                if (le1.value == null)
                                    retv = le2.value == null;
                                else 
                                    retv = le1.value.equals(le2.value);
                            } else {
                                if (p1 instanceof BinaryExpr
                                        && p2 instanceof BinaryExpr) {
                                    BinaryExpr be1 = (BinaryExpr) p1;
                                    BinaryExpr be2 = (BinaryExpr) p2;
                                    retv = (be1.op == be2.op)
                                            && isSame(be1.left, be2.left)
                                            && isSame(be1.right, be2.right);                                                                    
                                } else {
                                    if (p1.getTag() == TagConstants.LABELEXPR) {
                                        LabelExpr le1 = (LabelExpr) p1;
                                        LabelExpr le2 = (LabelExpr) p2;
                                        retv = le1.positive == le2.positive && le1.label == le2.label;
                                        retv = retv && isSame(le1.expr, le2.expr); 
                                    } else {
                                        retv = p1.equals(p2);
                                    }
                                }
                            }

                        }
                    }
                }
            }
        }


//        if (retv) {
//            System.out.println("isSame, are the same:");
//          printExpression(p1); System.out.print(": " + p1);  System.out.println(",");
//          printExpression(p2); System.out.print(": " + p2);  System.out.println();
//        }
//         System.out.println("--- Comparing: ");
//         printExpression(p1); System.out.println(",\n");
//         printExpression(p2); System.out.println("\n");
//         
//         if (retv) {
//            System.out.println("---- Same");
//        } else {
//            System.out.println("---- Different");
//        } 
//        

        return retv;
    }
    
   
    static /*@non_null*/ExprVec remove(/*@non_null*/ExprVec ev, /*@ non_null*/Expr e ) {
       ExprVec retv = ExprVec.make(ev.size());
       for (int i = 0; i < ev.size(); i++) {
           if (!isSame(ev.elementAt(i), e)) 
               retv.addElement(ev.elementAt(i));
       }
       return retv;
    }
    
    /*@ pure */public static int flaSize(/*@ non_null*/ASTNode node) {
        int size = 1;
        for (int i = 0; i < node.childCount(); i++) {
            Object child = node.childAt(i);

            if (child instanceof ASTNode) {
                size += flaSize((ASTNode) child);
            }
        }
        return size;
    }
    
    static /*@non_null*/Expr negSimple(/*@ non_null */Expr e) {
        if (isTrueLit(e))
            return GC.falselit;
        if (isFalseLit(e))
            return GC.truelit;
        
        return GC.not(e);
    }
    
    static /*@ non_null */Expr addFact(/*@ non_null*/Expr fact, /*@ non_null*/Expr e) {
        if (isSame(fact, e))
            return GC.truelit;
                    
        switch (e.getTag()) {
        case TagConstants.BOOLAND: {
            NaryExpr and = (NaryExpr) e;
            ExprVec conj = and.exprs;
            ExprVec retv = ExprVec.make(conj.size());
            for (int i = 0; i < conj.size(); i++) {
                Expr retvElement = addFact(fact, conj.elementAt(i));
                retv.addElement(retvElement);
            }
            return andSimple(retv);
        }
        case TagConstants.BOOLOR: {
            NaryExpr or = (NaryExpr) e;
            ExprVec oldVec = or.exprs;
            ExprVec disj = ExprVec.make(oldVec.size());
            for (int i = 0; i < oldVec.size(); i++) {
                Expr retvElement = addFact(fact, oldVec.elementAt(i));
                disj.addElement(retvElement);
            }
            return orSimple(disj);
        }
        case TagConstants.BOOLIMPLIES: {
            NaryExpr impl = (NaryExpr) e;
            Expr left = addFact(fact, impl.exprs.elementAt(0));
            Expr right = addFact(fact, impl.exprs.elementAt(1));            
            return implySimple(left, right);   
        }
        case TagConstants.BOOLNOT: {
            Expr neg = ((NaryExpr) e).exprs.elementAt(0);
            return negSimple(addFact(fact, neg));
        }
        case TagConstants.FORALL: {
            QuantifiedExpr qe = (QuantifiedExpr) e;
            if (!usesAnyVar(qe.vars, fact)) {
                Expr newBoundExpr = addFact(fact, qe.expr);
                Expr retvQE = recreateQuantifiedExprLazy(qe, newBoundExpr);
                return retvQE;
            } else {
                return e;
            }
        }
        case TagConstants.LABELEXPR: {
            LabelExpr le = (LabelExpr) e;
            return recreateLabelExprLazy(le, addFact(fact, le.expr));            
        }
        default: return e;
        }
        
    }
    
     
    public static Expr implySimple(Expr left, Expr right) {
        // (a ==> T) -> T
        if (isTrueLit(right))
            return GC.truelit;

        // (a ==> F) -> not(a)        
        if (isFalseLit(right)) {
            return negSimple(left);
        }

        // (F ==> a) -> T        
        if (isFalseLit(left))
            return GC.truelit;

        // (T ==> a) -> a
        if (isTrueLit(left))
            return right;

        return GC.implies(left, right);
    }
    
    static Expr pruneImplication(Expr left, Expr right) {
        ExprVec lList = flattenAnd(left);
        
        for (int i = 0; i < lList.size(); i++) {
            Expr f = lList.elementAt(i);
            right = addFact(f, right);
        }
        
        return implySimple(andSimple(lList), right);       
    }
    
    private static Expr pruneImplication(Expr expr) {
      if (expr.getTag() == TagConstants.BOOLIMPLIES) {
          ExprVec evec = ((NaryExpr) expr).exprs;
          return pruneImplication(evec.elementAt(0), evec.elementAt(1));
      } else
          return expr;  
    }
      

    public static Expr stripOffLabelsDeep(Expr e) {
        e = stripOffLabels(e);
        
        if (e instanceof NaryExpr) {
            NaryExpr ne = (NaryExpr) e;
            ExprVec oldVec = ne.exprs;
            ExprVec newVec = ExprVec.make(oldVec.size());
            boolean changed = false;
            for (int i = 0; i < oldVec.size(); i++) {
                Expr old_i = oldVec.elementAt(i);
                Expr new_i = stripOffLabelsDeep(old_i);
                newVec.addElement(new_i);
                changed = changed || (new_i != old_i);
             }

            if (changed) {
                return recreateNary(ne, newVec);
            } else {
                return ne;
            }
        }
        
        if (e instanceof QuantifiedExpr) {
            QuantifiedExpr qe = (QuantifiedExpr) e;
            return recreateQuantifiedExprLazy(qe, stripOffLabelsDeep(qe.expr));            
        }
        
        return e;
    }
    
    

    
    public static Expr factorAndFromOr(Expr o) {
        if (o.getTag() != TagConstants.BOOLOR)
            return o;
                
        NaryExpr or = (NaryExpr) o;
        ExprVec disj = or.exprs;
        if (disj.size() != 2)
            return o;
        
        ExprVec d0 = flattenAnd(disj.elementAt(0));
        ExprVec d1 = flattenAnd(disj.elementAt(1));
        ExprVec common = findCommon(d0, d1);
        
        if (common.size() > 0) {
            Expr orPart = GC.or(andSimple(difference(d0, common)), andSimple(difference(d1, common)));
            
            return GC.and(andSimple(common), normalizeOr(orPart));
        } else {
            return o;  
        }
        
        
    }
    
    static boolean isReflexiveBinary(int tag) {
        return (tag == TagConstants.ANYEQ) || 
               (tag == TagConstants.FLOATINGEQ) ||
               (tag == TagConstants.FLOATINGLE) ||
               (tag == TagConstants.INTEGRALEQ) ||
               (tag == TagConstants.INTEGRALLE) ||
               (tag == TagConstants.REFEQ);    
    }
    
    private static Expr normalizeIff(Expr e) {
        if (e.getTag() != TagConstants.BOOLEQ) 
            return e;
        
        NaryExpr ne = (NaryExpr) e;
        
        Expr e0 = ne.exprs.elementAt(0);
        Expr e1 = ne.exprs.elementAt(1);
        if (isSame(e0, e1)) {
            return GC.truelit;
        }
        
        if (isTrueLit(e0)) {
            return e1;
        }
        
        if (isTrueLit(e1)) {
            return e0;
        }
        
        return e;
    }
    
    private static Expr normalizeEq(Expr p) {
        int tag = p.getTag();
        if (isReflexiveBinary(tag)) {
            NaryExpr eq = (NaryExpr) p;
            ExprVec ev = eq.exprs;
            Expr e0 = ev.elementAt(0);
            Expr e1 = ev.elementAt(1);
            if (isSame(e0, e1))
                return GC.truelit;
         }
        
        return p;
    }
 
    private static Expr normalizeOr(Expr pred) {
        if (pred.getTag() != TagConstants.BOOLOR)
            return pred;
        
        ExprVec disj = removeDuplicates(((NaryExpr) pred).exprs);

        if (contains(disj, GC.truelit)) {
            return GC.truelit;
        }
        
        if (containsComplementary(disj)) {
            return GC.truelit;
        }
        
        disj = remove(disj, GC.falselit);
        
        return orSimple(disj);
    }
    
 
    private static Expr normalizeAnd(Expr e) {
        if (e.getTag() != TagConstants.BOOLAND)
            return e;

        //ExprVec conj = removeDuplicates(((NaryExpr) e).exprs);
        ExprVec conj = ((NaryExpr) e).exprs;
        
        if (contains(conj, GC.falselit))
            return GC.falselit;
        
        conj = remove(conj, GC.truelit);
        
//        if (containsComplementary(conj)) {
//            return GC.falselit;
//        }
        return andSimple(conj);
    }
    
    public static Expr pruneAnd(/*@ non_null*/Expr e) {
        ExprVec conjs = flattenAnd(e);
        
        for (int i = 0; i < conjs.size(); i++) {                        
            Expr conj_i = conjs.elementAt(i);
            
            //@ loop_invariant inner >= 0 & inner <= conjs.size();
            for (int inner = 0; inner < conjs.size(); inner++) {
                if (inner != i) {
                    Expr conj_inner = addFact(conj_i, conjs.elementAt(inner));
                    conjs.setElementAt(conj_inner, inner);
                }
            }
        }
        Expr retvDbg = andSimple(conjs);
        
        return retvDbg;
    }
    
    private static void debugTest(Expr origDbg, Expr retvDbg) {
        //        if (!isSame(origDbg, retvDbg)) {
        if (origDbg != retvDbg) {
            Expr implDbg = GC.implies(retvDbg, origDbg);           
            if (!Simplifier.runProver(implDbg)) {
                System.out.println("Problem  ");;
                //VcToString.computeTypeSpecific(origDbg, System.out);System.out.println();
                //VcToString.computeTypeSpecific(retvDbg, System.out);System.out.println();
                printExpression(origDbg);System.out.println();
                printExpression(retvDbg);System.out.println();
            } else {
                System.out.println("pruneAnd OK ");
            }             
        }        
    }
    
    static Expr normalize_deep(Expr e) {
       Expr origDbg = e;
        if (e.getTag() == TagConstants.LABELEXPR) {
            LabelExpr le = (LabelExpr) e;
            return recreateLabelExprLazy(le, normalize_deep(le.expr));
        }
        
        if (e instanceof NaryExpr) {
            NaryExpr oldNE = (NaryExpr) e;
            ExprVec oldVec = oldNE.exprs;
            ExprVec newVec = ExprVec.make(oldVec.size());
            boolean changed = false;
            for (int i = 0; i < oldVec.size(); i++) {
                Expr old_i = oldVec.elementAt(i);
                Expr new_i = normalize_deep(old_i);
                newVec.addElement(new_i);
                changed = changed || (new_i != old_i);
            }
            
            if (changed) { // optimize for identity
                e = recreateNary(oldNE, newVec);
            } 
        }
        
        if (e instanceof QuantifiedExpr) {
            QuantifiedExpr qe = (QuantifiedExpr) e;            
            e = recreateQuantifiedExprLazy(qe, normalize_deep(qe.expr));            
        }
 
        
        e = removeLetExpression(e);
//        e = normalizeOr(e);
        e = normalizeEq(e);
        
        e = normalizeIff(e);
        
        e = factorAndFromOr(e);
        
        e = pruneQuantifiedExpr(e);
        
        e = pruneAnd(e);
 //       e = normalizeAnd(e);
        
        e = pruneImplication(e);

//         System.out.println("+++orig -> new");        
//         debugTest(origDbg, e);
//         System.out.println("+++new -> orig");        
//         debugTest(e, origDbg);
        
        return e;
    }
    
    public static /*@ non_null*/Expr grind(/*@ non_null*/Expr pred) {     
        //opred = stripOffLabelsDeep(pred);
        
        Expr oldPred = pred;
        do {
            oldPred = pred;
            //printExpression("Before grind iter: ", pred);                     
            pred = normalize_deep(pred);
            //printExpression("Normalize: ", pred);
        } while (!isSame(oldPred, pred));
        return pred;
    }
    
    public static ExprVec findCommon(ExprVec e1, ExprVec e2) {
      ExprVec common = ExprVec.make();
      for (int i = 0; i < e1.size(); i++) {
          Expr ex1 = e1.elementAt(i);
          if (contains(e2, ex1))
              common.addElement(ex1);
      }
      return common;
    }
    
    
    /**
     * Removes elements that occur in both given lists from them.
     * @param e1
     * @param e2
     */
    public static /*@non_null@*/ExprVec removeCommon(ExprVec e1, ExprVec e2) {
        int i1 = 0;
        ExprVec common = ExprVec.make();
        while (i1 < e1.size()) {
            int i2 = 0;
            
            while (i2 < e2.size() && i1 < e1.size()) {
                Expr ex1 = e1.elementAt(i1);
                Expr ex2 = e2.elementAt(i2);
                if (isSame(ex1, ex2)) {
                    e1.removeElementAt(i1);
                    e2.removeElementAt(i2);
                    common.addElement(ex1);
                    i2 = 0;
                } else {
                    i2++;
                }
            }
            i1++;
        }
        return common;
    }
    
    private static void flattenAnd(Expr e, ExprVec monomial) {
        if (e.getTag() == TagConstants.BOOLAND) {
            NaryExpr ne = (NaryExpr) e;
            ExprVec ev = ne.exprs;
            for (int i = 0; i < ev.size(); i++)
                flattenAnd(ev.elementAt(i), monomial);
        } else {
            monomial.addElement(e);
        }
    }
    
    public static ExprVec flattenAnd(Expr e) {
        ExprVec varVec = ExprVec.make();
        flattenAnd(e, varVec);
        return varVec;
    }

    static ExprVec computeDependent(Expr e, ExprVec tested) {
        ExprVec depTest = ExprVec.make(1);
        depTest.addElement(e);

        ExprVec dependent = ExprVec.make();

        ExprVec toBeSplit = tested;

        while (depTest.size() > 0 && toBeSplit.size() > 0) {
            ExprVec currentDependent = ExprVec.make();
            ExprVec currentIndependent = ExprVec.make();

            dependencySplit(depTest, toBeSplit, currentDependent,
                    currentIndependent);

            dependent.append(currentDependent);
            toBeSplit = currentIndependent;
            depTest = currentDependent;
        }

        return dependent;
    }

    static void dependencySplit(ExprVec depTest, ExprVec initialVec,
            ExprVec dependent, ExprVec independent) {
        /*
         * reportln("== splitting ");
         * PredicateAbstraction.printlnExprVec(initialVec); reportln(" by ");
         * PredicateAbstraction.printlnExprVec(depTest);
         */

        for (int i = 0; i < initialVec.size(); i++) {
            Expr testedExpr = initialVec.elementAt(i);
            ExprVec addTo;
            if (shareVariables(depTest, testedExpr)) {
                addTo = dependent;
            } else {
                addTo = independent;
            }

            addTo.addElement(testedExpr);
        }
    }

    // boolean contains(GenericVarDecl varDec, GenericVarDeclVec vect) {
    // }

    
    public static Expr pruneQuantifiedExpr(Expr e) {
        if (e instanceof QuantifiedExpr) {
            QuantifiedExpr qe = (QuantifiedExpr) e;
            return pruneQuantifiedExpr(qe);
        }
        else
            return e;
    }
    /**
     * Removes quantifiers that are not used inside the quantified expression.
     */
    //@ requires qe != null;
    public static Expr pruneQuantifiedExpr(QuantifiedExpr qe) {
        Expr boundExpr = qe.expr;
        Expr rangeExpr = qe.rangeExpr;
      
        // get the relevant and invariant variables that is quantified over
        GenericVarDeclVec relevantVars = GenericVarDeclVec.make();
        GenericVarDeclVec nonRelevantVars = GenericVarDeclVec.make();

        AlgebraUtils.relevantVariablesSplit(qe, relevantVars, nonRelevantVars);

        // filter out the relevant pats and nopats
        ExprVec relevantPats = AlgebraUtils.filterIrelevantExpressions(qe.pats,
                nonRelevantVars);
        ExprVec relevantNoPats = AlgebraUtils.filterIrelevantExpressions(
                qe.nopats, nonRelevantVars);

        Expr retv;

        if (relevantVars.size() > 0) {
            // create a new quantified expr with only the relevant vars, if there are any
            retv = QuantifiedExpr.make(qe.sloc, qe.eloc, qe.quantifier,
                    relevantVars, rangeExpr, boundExpr, 
                    relevantNoPats, relevantPats);
        } else {
            // qe didn't quantify over anything relevant, strip off the
            // quantifier
            retv = boundExpr;
            if (qe.rangeExpr != null) {
                // if there was a range expression, translate it to an
                // implication
                retv = GC.implies(qe.rangeExpr, retv);
            }
        }

        return retv;
    } 
    
    
    /* ==== printing - debugging purposes ==== */

    public static void printExprVec(String s, ExprVec ev) {
      System.out.println(s);
      System.out.print('[');
      for (int i = 0; i < ev.size(); i++) {
          printExpression(ev.elementAt(i));
          System.out.print(i == ev.size() - 1 ? "]": ", " );
      }         
      System.out.println();
    }
    
    static String getDescr(int tag) {
        String op = null;
        switch (tag) {
        case TagConstants.BOOLIMPLIES:
            op = "IMPLIES";
            break;
        case TagConstants.BOOLAND:
        case TagConstants.BOOLANDX:
            op = "AND";
            break;
        case TagConstants.BOOLOR:
            op = "OR";
            break;
        case TagConstants.BOOLNOT:
            op = "NOT";
            break;
        case TagConstants.BOOLEQ:
            op = "IFF";
        }

        return op;
    }

    static String indent(int level) {
        String retv = "";
        for (int i = 0; i < level; i++) retv += ' ';
        return retv;
    }
    
    // only debugging purposes
    public static void printExpression(String s, Expr e) {
        System.out.println(s);
        printExpression(e);        
        System.out.println();
        System.out.flush();
    }

    // only debugging purposes
    public static void printExpression(Expr e) {
        printExpression(e, 0);        
    }

    public static void printExpression(Expr e, int level) {
        if ((getDescr(e.getTag()) != null)  && e instanceof NaryExpr) {
            System.out.println(indent(level) + '(' + getDescr(e.getTag()));
            
            ExprVec exprs = ((NaryExpr) e).exprs;
            for (int i = 0; i < exprs.size(); i++) {
                printExpression(exprs.elementAt(i), level+1);
                if (i == exprs.size() - 1) {
                    System.out.print(')');            
                }
                else {
                    System.out.println();
                }
            }            
      }  else {
          int tag = e.getTag();
          if (tag == TagConstants.FORALL || tag == TagConstants.EXISTS) {
              QuantifiedExpr qe = (QuantifiedExpr) e;
              String descr = tag ==TagConstants.FORALL ? "FORALL" : "EXISTS";
              System.out.print( indent(level) + '(' + descr + " (");
              for (int i = 0; i < qe.vars.size(); i++) {
                  System.out.print(Atom.printableVersion(UniqName.variable(qe.vars.elementAt(i))));
                  if (i == qe.vars.size() -1) {
                      System.out.println(");");
                  } else {
                      System.out.print(' ');
                  }                  
              }
 /*             if (qe.rangeExpr != null && !isTrueLit(qe.expr)) {
                  printExpression(qe.rangeExpr, level + 1); System.out.print(';');
              } */
              printExpression(qe.expr, level + 1);
              System.out.print(')');
          } else {    
             System.out.print(indent(level));
             VcToString.computeTypeSpecific(e, System.out);
          }
      }        
    }

    

}
