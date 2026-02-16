/**
 * @title Spec Tester
 * @description Creates tests for method specifications
 * @author Mikolas Janota
 */


package escjava.dfa.daganalysis;

import javafe.ast.*;
import escjava.ast.GuardedCmd;
import escjava.translate.InitialState;
import javafe.tc.TypeSig;
import escjava.sp.DSA;

 // these are needed when creating assert false
import escjava.ast.ExprStmtPragma;
import escjava.ast.LabelExpr;
import escjava.translate.GC;

import javafe.util.Location;

// debug imports
import escjava.ast.EscPrettyPrint; 



/**
 * A class that encapsulated the function of constructing a test for a method's spec.
 * All this is quite experimental at the moment and the code below should be refactored (besides other things).
 *
 * @author Mikolas Janota
 */
public class SpecTester {

    /*@pure*/ public static boolean knowHowToCheck(RoutineDecl r) {
        return (r instanceof MethodDecl) || (r instanceof ConstructorDecl);
    }

    private static EvalStmt routineCall(MethodDecl r, ExprVec callArgs) {
        ThisExpr thisExpr = ThisExpr.make(null, r.getEndLoc());
        ExprObjectDesignator thisOD = ExprObjectDesignator.make(r.getEndLoc(), thisExpr);
        MethodInvocation mi = MethodInvocation.make(thisOD, r.id(), null, r.getEndLoc(), r.getEndLoc(), callArgs);
        mi.decl = r;
        return EvalStmt.make(mi);
    }

    private static EvalStmt routineCall(ConstructorDecl r, ExprVec callArgs) {
        SimpleName className = SimpleName.make(r.parent.id, r.getEndLoc());
        TypeName classTypeName = TypeName.make(className);

        TypeSig.setSig(classTypeName, TypeSig.getSig(r.parent));

        NewInstanceExpr ni = NewInstanceExpr.make(null, r.getEndLoc(), classTypeName, callArgs, null, r.getEndLoc(), r.getEndLoc());
        ni.decl = r;
        return EvalStmt.make(ni);
    }

    /**
     * Construct a test for the given method.
     */
    public static  MethodDecl fabricateTest(RoutineDecl r, TypeSig sig, InitialState initState) {
        StmtVec statements = StmtVec.make(); // statements in the body of the fabricated method

        // declare variables that will serve as arugments of the called method
        // the argument vector is created as well
        ExprVec callArgs = ExprVec.make();
        for (int i = 0;  i < r.args.size(); i++ ) {
            FormalParaDecl arg = r.args.elementAt(i);
            LocalVarDecl ti = LocalVarDecl.make(0, ModifierPragmaVec.make(),
                                                Identifier.intern("mj_" + arg.id), arg.type,
                                                r.getEndLoc(),
                                                null,
                                                0);
            statements.addElement(VarDeclStmt.make(ti)); // declare var in statements
            callArgs.addElement(VariableAccess.make(ti.id, ti.locId, ti)); // add to args
        }

        // create a statmement that calls the tested routine
        EvalStmt call = null;
        if (r instanceof MethodDecl) {
            call = routineCall((MethodDecl)r, callArgs);
        }
        if (r instanceof ConstructorDecl) {
            call = routineCall((ConstructorDecl)r, callArgs);
        }



        // add call to the statements
        statements.addElement(call); 


        // stick and assert false after the call
        statements.addElement(createAssertFalse(r.getEndLoc(), "after_call_assert"));

        // create the whole testing method
        BlockStmt body = BlockStmt.make(statements, r.getEndLoc(), r.getEndLoc());
        MethodDecl fr =  MethodDecl.make(4194304,
                                         null, 
                                         null, 
                                         FormalParaDeclVec.make(), 
                                         TypeNameVec.make(),
                                         body, 
                                         r.getEndLoc(), r.getEndLoc(), r.getEndLoc(), r.getEndLoc(), 
                                         Identifier.intern(r.id() + "_testingMethod"),
                                         JavafePrimitiveType.make(ASTTagConstants.VOIDTYPE, r.getEndLoc()),
                                         r.getEndLoc());

        // debug
        //    System.err.println("fabricated method: " + fr);

        fr.setParent(r.getParent()); // put the method in the same class as the tested routine
        return fr;
    }



    /**
     * Run reachability analysis on the test created by <code>fabricateTest</code>.
     *
     * @param gc a guarded command, gets turned to DSA 
     */
    public static void runReachability(GuardedCmd gc, int loc) {
        if (escjava.Main.options().pgc) {
            System.out.println("\n *** Fabricated GC ");
            ((EscPrettyPrint)PrettyPrint.inst).print(System.out, 0, gc);
        }

        GuardedCmd bodyDSA = DSA.dsa(gc);

        //        LabelExpr le = LabelExpr.make(loc, loc, false, Identifier.intern("Assert@" + Location.toLineNumber(loc) + "." + Location.toColumn(loc)), GC.falselit);
        //        GuardedCmd assertFalse = escjava.ast.ExprCmd.make(escjava.ast.TagConstants.ASSERTCMD, le, javafe.util.Location.NULL);
        //        bodyDSA = GC.seq(bodyDSA, assertFalse);

        // debug
        if (escjava.Main.options().pdsa) {
            System.out.println("\n ****  DSA  Fabricated for the test ");
            ((EscPrettyPrint)PrettyPrint.inst).print(System.out, 0, bodyDSA);
            System.out.println();
        }


        // run reachability analyzis on the resulting DSA
        ReachabilityAnalysis.analyze(bodyDSA);
        //processRoutineDecl(fr, sig, initState);
    }

    private static  ExprStmtPragma createAssertFalse(int loc, String label) {
        LiteralExpr falseLit = LiteralExpr.make(ASTTagConstants.BOOLEANLIT, Boolean.FALSE, loc);
        javafe.tc.FlowInsensitiveChecks.setType(falseLit, JavafePrimitiveType.make(ASTTagConstants.BOOLEANTYPE, loc));
        LabelExpr le = LabelExpr.make(loc, loc, false, Identifier.intern(label),  falseLit);

        return  ExprStmtPragma.make(escjava.ast.TagConstants.ASSERT, le, null, loc);
    }

}