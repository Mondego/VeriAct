package escjava.dfa.daganalysis;

import java.util.Properties;

import javafe.ast.Expr;
import javafe.ast.ExprVec;
import javafe.ast.PrettyPrint;
import javafe.ast.Util;
import escjava.ProverManager;
import escjava.ast.EscPrettyPrint;
import escjava.ast.GuardedCmd;
import escjava.translate.GC;
import escjava.translate.VcToString;

public class Simplifier {

    // TODO: make this configurable from the command line
    private static int pushedCnt = 0;
    
    public static void push(Expr e) {
        ++pushedCnt;
        ProverManager.push(e);
    }
    
    public static void popAll() {
        while (pushedCnt-- != 0) ProverManager.pop();
    }

    public static Expr simplify(Expr e) {
        return (Expr) e.clone();
    }


    public static boolean isFalse(Expr e) {
        // optimize for boolean literals
        if (GC.isBooleanLiteral(e, true))
            return false;
        if (GC.isBooleanLiteral(e, false))
            return true;

        // negate
        Expr neg = GC.not(e);

        /* DBG
        printExpr("Running prover on:", e);
        System.out.println();*/

        // run prover on the negated formula
        return runProver(neg);
    }


    /**
     * Returns true only if [e] is valid.
     * @param e The expression whose validity is checked.
     * @return true only if [e] is valid
     */
    public static boolean runProver(Expr e) {
        Properties props = new Properties();
        
        // flag for all provers
        props.setProperty("TimeLimit", "60");
        
        // flags for Fx7
        //props.setProperty("max_quant_iters", "500");
        //props.setProperty("max_main_iters", "500");
        
        //System.err.println("vc_size " + Util.size(e));
        TimeUtil.start("prover_time");
        boolean ans = ProverManager.isValid(e, props);
        //System.err.println(ans ? "valid" : "invalid");
        TimeUtil.stop("prover_time");
        return ans;
    }

    /* =========== DBG ============ */
    public static void printExpr(String s, Expr e) {
        System.out.println(s);
        VcToString.computeTypeSpecific(e, System.out);
        System.out.println();
    }

    public static void printGC(GuardedCmd gc) {
        ((EscPrettyPrint) PrettyPrint.inst).print(System.out, 0, gc);
        System.out.println();
    }

    public static void printlnExpr(Expr e) {
        VcToString.computeTypeSpecific(e, System.out);
        System.out.println();
    }

    public static void printlnExprVec(ExprVec evec) {
        System.out.print("[");
        for (int i = 0; i < evec.size(); i++) {
            printlnExpr(evec.elementAt(i));
            System.out.print(",");
        }
        System.out.println("]");
    }
}
