/**
 * @title Reachability Analysis
 * @description Reachability Analysis
 * @author Radu Grigore, Michal Moskal and Mikolas Janota
 * @copyright 2007, Mobius IST-15905
 * @license MIT/X11
 * @version "$Id$"
 */

package escjava.dfa.daganalysis;

import java.util.Hashtable;

import javafe.ast.Expr;
import javafe.ast.GenericVarDecl;
import javafe.ast.GenericVarDeclVec;
import javafe.ast.LocalVarDecl;
import javafe.ast.PrettyPrint;
import javafe.ast.VariableAccess;
import javafe.util.Assert;
import javafe.util.Location;
import escjava.ast.AssignCmd;
import escjava.ast.EscPrettyPrint;
import escjava.ast.ExprCmd;
import escjava.ast.GuardedCmd;
import escjava.ast.SubGetsCmd;
import escjava.ast.SubSubGetsCmd;
import escjava.ast.TagConstants;
import escjava.dfa.cfd.CFD;
import escjava.dfa.cfd.CodeNode;
import escjava.dfa.cfd.CouplingNode;
import escjava.dfa.cfd.ExceptionNode;
import escjava.dfa.cfd.Node;
import escjava.dfa.cfd.NodeList;
import escjava.translate.GC;
import escjava.translate.UniqName;

public class DAGBackpropagation {
  
    public static final String INPUT_GRAPH_NOT_ACYCLIC = "The input graph is not acyclic";
    public static final String INPUT_GRAPH_NOT_CONSISTENT = "The input graph is not consistent";

    Hashtable nodeToInfo;

    /**
     * The control-flow diagram on which the analysis is performed.
     */
    CFD cfd;

    /**
     * The predicate that is back-propagated.
     */
    Expr reachPredicate;

    /**
     * The back-propagation will lead to this node.
     */
    Node start;

    /**
     * Back-propagation will start from this node.
     */
    Node reach;

    //@ private behavior
    //@   ensures	this.cfd == cfd;
    //@   ensures this.reachPredicate == reachPredicate;
    //@   ensures this.start == start;
    //@   ensures this.reach == reach;
    public DAGBackpropagation(CFD cfd) {
        this.cfd = cfd;
        Assert.notFalse(cfd.isConsistent(), INPUT_GRAPH_NOT_CONSISTENT);
        Assert.notFalse(cfd.isAcyclic(), INPUT_GRAPH_NOT_ACYCLIC);
    }

    /**
     * Computes the back-propagation.
     */
    public void backPropagate(Expr reachPredicate, Node start, Node reach) {
        this.reachPredicate = reachPredicate;
        this.start = start;
        this.reach = reach;

        this.initBackProp();
        this.computeIntersections();
        this.backProp();
    }

    /**
     * After the back-propagation was computed returns the computed information for a given node.
     * @param node the node for which the information was requested
     * @return the reach-expression for <code>node</code>
     */
    public /*@ non_null @*/ Expr getInformation(/*@ non_null @*/ Node node) {
        Info i = this.getInfo(node);
        if (i != null)
            return i.getExpr();
        else
            return GC.falselit; // for non-reachable nodes
    }

    /**
     * Initializes the resources needed for back-propagation.
     */
    //@ ensures this.nodeToInfo != null;
    void initBackProp() {
        if (this.nodeToInfo == null)
            this.nodeToInfo = new Hashtable();
        else
            this.nodeToInfo.clear();

        Info reachInfo = new Info(0, this.reachPredicate);
        this.addInfo(this.reach, reachInfo);
    }


    /**
     * Computes the DP values.
     */
    void computeIntersections() {
        this.addDefaultInfo(this.start);
        this.findPath(this.start, this.reach);
    }

    boolean findPath(Node from, Node to) {
        if (from == to)
            return true; // we have reached the goal

        Info fromInfo = this.getInfo(from);

        boolean retv = false;
        NodeList children = from.getChildren();
        for (NodeList.Enumeration e = children.elements(); e.hasMoreElements();) {
            Node child = e.nextElement();
            Info childInfo = this.getInfo(child);
            if (childInfo == null) {
                this.addDefaultInfo(child);
                childInfo = this.getInfo(child);
            }

            if (childInfo.getDP() > 0 || findPath(child, to)) {
                fromInfo.incDP();
                retv = true;
            }
        }

        return retv;
    }

    /**
     * Back-propagates the information between adjacent nodes.
     *
     * @param from the node from which information is propagated from
     * @param to the to which information is propagated to
     */
    protected void backPropAdjacent(Node from, Node to) {
        Expr info = null;
        Expr fromExpr = this.getInfo(from).getExpr();
        if ((to instanceof CouplingNode) || (to instanceof ExceptionNode)) {
            info = fromExpr;
        } else {
            if (to instanceof CodeNode) {
                CodeNode cn = (CodeNode) to;
                info = backPropAdjacentCodeNode(fromExpr, cn);

            } else {
                //@ unreachable;
                Assert.fail("Fall thru on " + to);
            }
        }

         info = variableClosure(info, to.getScope(), from.getScope());
         this.getInfo(to).andExprWith(info);
    }

    Expr variableClosure(Expr expression, GenericVarDeclVec newScope, GenericVarDeclVec oldScope) {
        GenericVarDeclVec enclVars = AlgebraUtils.difference(oldScope, newScope);

        Expr closedExpression = (Expr) expression.clone();

        for (int i = 0; i < enclVars.size(); i++) {
            GenericVarDecl var = enclVars.elementAt(i);
            if (AlgebraUtils.usesVar(var, expression)) {
                LocalVarDecl boundVar = UniqName.newBoundVariable(var.id.toString() + "_scope");
                Expr boundVarAccess = VariableAccess.make(boundVar.id, var.locId, boundVar); 
                closedExpression = GC.subst(var, boundVarAccess, closedExpression);
                closedExpression = GC.forall(boundVar, closedExpression);
            }
        }

        return closedExpression;
    }


    /**
     * Back-propagates the information between adjacent nodes, where the node
     * <code>to</code> is a <code>CodeNode</code>.
     *
     * @param expression the expression to propagate
     * @param to the to which information is propagated to
     */
    //@ requires expression != null;
    //@ requires to != null;
    //@ ensures \result != null;
    Expr backPropAdjacentCodeNode(Expr expression, CodeNode to) {
        GuardedCmd command = to.getCode();

        Expr info;
        if (isNoop(command)) {// the command has no effect, simply copy the information
            info = expression;
        }
        else {// compute the weakest precondition otherwise
            int tag = command.getTag();
           
            
            // convert asserts to assumes
            if (tag == TagConstants.ASSERTCMD) {
                command = assertToAssume((ExprCmd) command);
            }
            
            if (tag == TagConstants.ASSUME) {
                ExprCmd ecmd = (ExprCmd) command;
                ecmd = stripOffLabelsCommand((ExprCmd) command);
                Expr commandPred = ecmd.pred;
                if (!AlgebraUtils.shareVariables(commandPred, expression)) {
                    return expression; // skip expressions that do not share variables with the back propagated one
                }
            }
                     
            info = wp(command, expression);
	        //info = AlgebraUtils.grind(info);
        }

        return info;
    }


    /**
     * Computes normal behaviour - weakest precondtion for a given command and
     * postcondition.
     */
    private static Expr wp(GuardedCmd command, Expr normalPost) {
        if (command instanceof AssignCmd) {
            AssignCmd assignCmd = (AssignCmd) command;
            return wpAssignCommand(assignCmd, normalPost);
        } else {
            int tag = command.getTag();
            switch (tag) {
            case TagConstants.ASSUMECMD:
                Expr pred = ((ExprCmd) command).pred;
                return GC.implies(pred, normalPost);
            default:
                ((EscPrettyPrint) PrettyPrint.inst).print(System.out, 0, command);
                System.out.println(); System.out.println();
                
                
                Assert.fail("Unexpected command " + command);
                return null;
            }

        }
    }

    public static Expr wpAssignCommand(AssignCmd assignCmd, Expr normal) {
        int cmdTag = assignCmd.getTag();

        // Calculate the rhs expression
        Expr nuv;
        switch(cmdTag) {
        case TagConstants.GETSCMD:
        case TagConstants.RESTOREFROMCMD:
            nuv = assignCmd.rhs;
            break;
        case TagConstants.SUBGETSCMD:
            // store(v,i,x)

            SubGetsCmd sgc = (SubGetsCmd)assignCmd;
            nuv = GC.nary(Location.NULL, Location.NULL,
                          TagConstants.STORE, sgc.v, sgc.index, sgc.rhs);
            break;
        case TagConstants.SUBSUBGETSCMD:
            // store(v, i1, store(select(v,i1), i2, x))
            SubSubGetsCmd ssgc = (SubSubGetsCmd)assignCmd;

            Expr innermap = GC.nary(Location.NULL, Location.NULL,
                                    TagConstants.SELECT,
                                    ssgc.v, ssgc.index1);
            Expr newinnermap = GC.nary(Location.NULL, Location.NULL,
                                       TagConstants.STORE,
                                       innermap, ssgc.index2, ssgc.rhs);
            nuv = GC.nary(Location.NULL, Location.NULL,
                          TagConstants.STORE,
                          ssgc.v, ssgc.index1, newinnermap);
            break;
        default:
            Assert.fail("Unreachable");
            nuv = null; // dummy assignment
        }

        GenericVarDecl vv  = assignCmd.v.decl;

        Expr wpNormal = GC.subst(vv, nuv, normal);
        return wpNormal;
    }

    static ExprCmd stripOffLabelsCommand(ExprCmd command) {
        Expr pred = command.pred;
        Expr stripped = AlgebraUtils.stripOffLabels(pred);
        return ExprCmd.make(command.cmd, stripped, command.loc);
    }

    static ExprCmd assertToAssume(ExprCmd assertCmd) {
        Expr pred = assertCmd.pred;
        return ExprCmd.make(TagConstants.ASSUMECMD, pred, assertCmd.loc);        
    }

    /**
     * Decides whether a given command has any effect on the reasoning.
     * @param command
     *      the command for which we want to know whether it has any effect
     * @return
     *     <code>true</code>, if  and only if the command doesn't have any effect on the reasoning
     */
    boolean isNoop(GuardedCmd command) {
        int tag = command.getTag();
        switch (tag) {
        case TagConstants.ASSERTCMD:
        case TagConstants.ASSUMECMD:
            ExprCmd ec = (ExprCmd)command;
            Expr pred = ec.pred;
            return AlgebraUtils.isTrueLit(pred);
        case TagConstants.SKIPCMD:
            return true;
        default:
            return false;
        }
    }


    /**
     * Performs the back-propagation.
     */
    void backProp() {
        DPNodeList todoList = new DPNodeList();
        Node bp_node = this.reach;

        do {
            NodeList parents = bp_node.getParents();

            // explore the parents of the bp_node node
            for (NodeList.Enumeration e = parents.elements(); e.hasMoreElements();) {
                Node parent = e.nextElement();
                Info parentInfo = this.getInfo(parent);

                if (parentInfo != null && parentInfo.getDP() > 0) {
                    this.backPropAdjacent(bp_node, parent);
                    parentInfo.decDP();
                    if (!todoList.member(parent))
                        todoList.addNode(parent);
                }
            }

            bp_node = todoList.pickReadyNode();
        } while (bp_node != null);
    }

    void addInfo(/*@ non_null @*/Node n, /*@ non_null @*/Info i) {
        this.nodeToInfo.put(n, i);
    }

    void addDefaultInfo(/*@ non_null @*/Node n) {
        Info i = new Info(0, GC.truelit);
        //        Info i = new Info(0, GC.falselit);
        this.addInfo(n, i);
    }

    Info getInfo(/*@ non_null @*/Node n) {
        return (Info) this.nodeToInfo.get(n);
    }

    class DPNodeList extends NodeList {

        int findReadyNode() {
            for (int i = 0; i < count; i++) {
                Info ni = getInfo(nodes[i]);
                if (ni.getDP() == 0) {
                    return i;
                }
            }

            return -1;
        }

        public Node pickReadyNode() {
            if (this.empty())
                return null;

            int readyNodeIndex = this.findReadyNode();
            Node retv = this.nodes[readyNodeIndex];
            this.nodes[readyNodeIndex] = this.nodes[count - 1];
            count--;

            return retv;
        }
    }

    class Info {
        int dp;

        Expr pred;

        public Info(int dp, Expr pred) {
            this.dp = dp;
            this.pred = pred;
        }

        public void incDP() {
            this.dp++;
        }

        public void decDP() {
            this.dp--;
        }

        public void andExprWith(Expr with) {
            this.pred = AlgebraUtils.and(this.pred, with);
        }
	/*
        public void orExprWith(Expr with) {
            this.pred = GC.or(this.pred, with);
	    }*/


        public int getDP() {
            return this.dp;
        }

        public Expr getExpr() {
            return this.pred;
        }

    }

}
