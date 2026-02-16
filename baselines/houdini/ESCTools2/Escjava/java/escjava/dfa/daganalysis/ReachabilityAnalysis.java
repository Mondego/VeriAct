/**
 * @title Reachability Analysis
 * @description Reachability Analysis
 * @author Radu Grigore, Michal Moskal and Mikolas Janota
 * @copyright 2007, Mobius IST-15905 
 * @license MIT/X11
 * @version "$Id$"
 */


package escjava.dfa.daganalysis;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import javafe.ast.Expr;
import javafe.ast.ExprVec;
import javafe.ast.Util;
import javafe.ast.VariableAccess;
import javafe.util.Assert;
import javafe.util.ErrorSet;
import javafe.util.Location;
import escjava.ast.ExprCmd;
import escjava.ast.GuardedCmd;
import escjava.ast.LabelExpr;
import escjava.ast.NaryExpr;
import escjava.ast.TagConstants;
import escjava.dfa.buildcfd.GCtoCFDBuilder;
import escjava.dfa.cfd.CFD;
import escjava.dfa.cfd.CodeNode;
import escjava.dfa.cfd.Node;
import escjava.dfa.cfd.NodeList;
import escjava.dfa.cfd.NodeList.Enumeration;
import escjava.translate.GC;
import escjava.translate.LabelData;
import escjava.translate.VcToString;

/**
 * Used to sort labels such as those corresponding to postcondition
 * checks come first and otherwise are sorted by file line number;
 * also helps to extract a location---the most significant one---
 * from the label.
 * 
 * TODO: All these labels look horrible. Why both LabelData and ints? etc.
 * TODO: Make static all methods that can be statics so that |this| is
 *       not carried around.
 * 
 * @author mikolas rgrig
 */
class LabelDataUtil implements Comparator {
    
    // Message strings
    public static final String DUPLICATE_ERROR = "Hmm? This is strange: same error at the same location?";
    public static final String LABEL_WITHOUT_LOCATION = "We expect that each label has a location.";
    
    /** Extract most significant location. */
    static public int getLocation(LabelData ld) {
        if (ld.hasUseLoc()) {
            return ld.getUseLoc();
        } else if (ld.hasDeclLoc()) {
            return ld.getDeclLoc();
        }
        Assert.fail(LABEL_WITHOUT_LOCATION);
        // If you are really sure you want label without location within ESC/Java
        // then for this class it should be fine if you just remove the assertion above.
        return Location.NULL;
    }
    
    public int compare(Object ao, Object bo) {
        LabelData a = (LabelData)ao;
        LabelData b = (LabelData)bo;
        int locA = getLocation(a);
        int locB = getLocation(b);
        int lineA = Location.toLineNumber(locA);
        int lineB = Location.toLineNumber(locB);
        int tagA = a.getMsgTag();
        int tagB = b.getMsgTag();
        boolean postA = isPost(tagA); 
        boolean postB = isPost(tagB);
        
        if (postA && postB) return tagA - tagB;
        if (postA != postB) return postA ? -1 : +1;
        if (lineA != lineB) return lineA - lineB;
        if (locA != locB) return locA - locB;
        if (tagA != tagB) return tagA - tagB;
        Assert.fail(DUPLICATE_ERROR);
        return -1;
    }
    
    /** Tags from (exceptional) postconditions. */
    private static final int[] POST_TAGS = {
        TagConstants.CHKPOSTCONDITION,
        TagConstants.CHKUNEXPECTEDEXCEPTION,
        TagConstants.CHKUNEXPECTEDEXCEPTION2
    };
    private static boolean isPost(int tag) {
        int i;
        for (i = 0; i < POST_TAGS.length && tag != POST_TAGS[i]; ++i);
        return i < POST_TAGS.length;
    }
}

/** A simple structure to associate label information to nodes. */
class NodeAndLabel {
    public Node node;
    public LabelData ld;
    public int loc;
    public NodeAndLabel(Node n) {
        node = n;
        ld = null;
        loc = Location.NULL;
    }
    public NodeAndLabel(Node n, String label) {
        Assert.notFalse(label != null);
        ld = LabelData.parse(label);
        loc = LabelDataUtil.getLocation(ld);
        node = n;
    }
}


/**
 * Finds commands that are unreachable. Example:
 * <tt>
 *   if (x > 1) {
 *     // ... code that does not modify x ... 
 *     if (x <= 1) {
 *       // unreachable code
 *     }
 *   }
 * </tt>
 */
public class ReachabilityAnalysis {
    // the control flow graph, its nodes, and those we are interested in
    private CFD graph;
    private HashSet graphNodes;

    // strongest preconditions
    private Map spOfNode;
    
    
    /*
     * The DAG-to-tree transformation. Tree is an euphemism because only
     * sharing that shrinks the size of the printed version is eliminated.
     * 
     * We maintain three maps
     *   dag -> var -> tree -> size
     * of type
     *   NaryExpr -> VariableAccess -> NaryExpr -> Integer
     * These are used to cache partial results. Note that some unnecessary
     * plucking is done because of this caching. I could clear the caches
     * between two invocations of dagToTree if I really want to keep the
     * size to a minimum.
     */
    
    // variables used for transforming the VC dag into a VC (almost-)tree
    private HashMap dagToVarCache; // (NaryExpr->VariableAccess)
    private HashMap varToTreeCache; // (VariableAccess->NaryExpr)
    private HashMap treeToSizeCache; // (NaryExpr->Integer)
    private HashMap parentsCnt; // counts parents for VC nodes (NaryExpr->Integer)
    private HashSet seenExprs; // used to limit various DAG traversals
    private HashSet usedVariables; // used to collect the variables used by a tree
    private static final String NAME = "ratmpvar"; // used to name variables
    private int varNameCnt; // to make the variable names unique
    
    /* When this is 1 all sharing that makes VCs bigger is eliminated.
     * Bigger values leave some sharing in therefore resulting in slightly
     * bigger VCs. However, they _might_ be processed quicker.
     * TODO: perhaps remove this parameter
     */
    private static final int THRESHOLD = 0;
    
    // Maximum VC size
    private static final int VC_LIMIT = 200000;
    
    // Set when a caution should be issued
    private boolean vcTooBig;
    
    
    // These are used to represent the compressed graph of the labels.
    private HashMap/*<Node,Integer>*/ nodeToLabelCache;
    private NodeAndLabel[] labels; // identified later on by indices in this array
    private Expr[] preconditions; // maps labels to their preconditions
    private int[] labelState; // unknown, sat, or unsat
    private int[][] labelChildren;
    private int[][] labelParents;
    private int[] dominator; // the immediate dominators
    private int[][] dominated; // the nodes immediately dominated
    private int[] dominatedCnt; // used during the construction of the dominator tree
    
    // used by various DFS-like algorithms
    private boolean[] labelTag; // is thelabel processed
    private int[] labelsSort; // topological sort of the labels
    private int[] labelsInvSort; // inverse of labelsSort
    private int labelOrder; // used during construction

    // Auxiliar information used during the construction of the above
    // NOTE: in a sane programming language I would just use these but
    //       I just hate casting and (un)boxing stuff all over the place.
    private ArrayList/*<NodeAndLabel>*/ labelsTmp;
    private ArrayList/*<ArrayList<Integer>>*/ labelParentsTmp;
    private ArrayList/*<ArrayList<Integer>>*/ labelChildrenTmp;
    private HashMap/*<HashSet<Integer>>*/ collectNodesCache;
    
    
    // NOTE that automatic provers are incomplete and sometimes they will
    // say invalid (interpreted here as SAT) even if the formula is actually
    // unsatisfiable.  We'll make sure we never set the state UNSAT for
    // situations where we don't really know. This way we wan't warn when there
    // really isn't a problem.
    private static final int UNKNOWN = 0; // the prover was not asked or it didn't know
    private static final int SAT = 1;  // satisfiable, the program can reach the label
    private static final int UNSAT = 2; // unsatifiable, we should warn
    private static final int TOOBIG = 3; // asking the prover would probably take too long
    
    private String assertLabel(GuardedCmd gc) {
        if (gc.getTag() == TagConstants.ASSERTCMD) {
            ExprCmd ec = (ExprCmd) gc;
            Expr e = ec.pred;
            if (e.getTag() == TagConstants.LABELEXPR) {

                LabelExpr le = (LabelExpr) e;
                String label = le.label.toString();

                return label;
            }
        }
        return null;
    }
    
    /**
     * Returns -1 if |n| is not an interesting node. Otherwise it
     * returns its index. Uses |nodeToLabelCache| to cache the results. 
     */
    private int nodeToLabelIdx(Node n) {
        Integer idx = (Integer)nodeToLabelCache.get(n);
        if (idx != null) return idx.intValue();
        
        int result = -1;
        String label = getLabel(n);
        if (label != null) {
            NodeAndLabel nl = new NodeAndLabel(n, label);
            if (nl.loc != Location.NULL) {
                result = labelsTmp.size();
                labelsTmp.add(nl);
            }
        }
        nodeToLabelCache.put(n, new Integer(result));
        labelParentsTmp.add(new ArrayList());
        labelChildrenTmp.add(new ArrayList());
        return result;
    }

    private void recSortLabels(int l) {
        if (labelTag[l]) return;
        labelTag[l] = true;
        for (int i = 0; i < labelChildren[l].length; ++i)
            recSortLabels(labelChildren[l][i]);
        labelsSort[labelOrder] = l;
        labelsInvSort[l] = labelOrder--;
    }
    
    private void sortLabels() {
        labelsSort = new int[labels.length];
        labelsInvSort = new int[labels.length];
        labelTag = new boolean[labels.length];
        labelOrder = labels.length - 1;
        recSortLabels(0);
    }
    
    // Adds all nodes reachable from [n] (children) to [graphNodes].
    // Also collects the interesting nodes.
    private void recCollectNodes(Node n, int labeledParent) {
        // return if already called with these parameters
        HashSet/*<Integer>*/ seenParents = (HashSet)collectNodesCache.get(n);
        if (seenParents == null) {
            seenParents = new HashSet();
            collectNodesCache.put(n, seenParents);
        }
        if (seenParents.contains(new Integer(labeledParent))) return;
        seenParents.add(new Integer(labeledParent));
        
        // do stuff
        int thisLabel = nodeToLabelIdx(n);
        if (thisLabel != -1) labeledParent = thisLabel;
        graphNodes.add(n);
        ArrayList/*<Integer>*/ children = (ArrayList)labelChildrenTmp.get(labeledParent);
        
        Enumeration c = n.getChildren().elements();
        while (c.hasMoreElements()) {
            Node child = c.nextElement();
            int childLabel = nodeToLabelIdx(child);
            if (childLabel != -1) {
                ArrayList/*<Integer>*/ parents = (ArrayList)labelParentsTmp.get(childLabel);
                parents.add(new Integer(labeledParent));
                children.add(new Integer(childLabel));
            }
            recCollectNodes(child, labeledParent);
        }
    }
    
    private int[] toIntArray(Collection/*<Integer>*/ al) {
        int[] result = new int[al.size()];
        Iterator it = al.iterator();
        int i = 0;
        while (it.hasNext())
            result[i++] = ((Integer)it.next()).intValue(); 
        return result;
    }
    
    /**
     * Collects all nodes reachable from the initial node of |graph| into
     * |graphNodes|. It also constructs a smaller graphs that contains only
     * the labeled (i.e., interesting) nodes.
     */
    private void collectNodes() {
        TimeUtil.start("collect_nodes_time");
        collectNodesCache = new HashMap();
        graphNodes = new HashSet();
        nodeToLabelCache = new HashMap/*<Node,Integer>*/();
        labelsTmp = new ArrayList();
        labelParentsTmp = new ArrayList();
        labelChildrenTmp = new ArrayList();
        Node init = graph.getInitNode();
        String label = getLabel(init);
        NodeAndLabel nl = label == null ? 
            new NodeAndLabel(init) : new NodeAndLabel(init, label);
        labelsTmp.add(nl);
        nodeToLabelCache.put(init, new Integer(0));
        labelParentsTmp.add(new ArrayList());
        labelChildrenTmp.add(new ArrayList());
        recCollectNodes(init, 0);
        
        // Convert ArrayLists to simple arrays. (I wouldn't do this in a later
        // version of Java.)
        labels = (NodeAndLabel[])labelsTmp.toArray(new NodeAndLabel[0]);
        preconditions = new Expr[labels.length];
        labelParents = new int[labels.length][];
        labelChildren = new int[labels.length][];
        for (int i = 0; i < labels.length; ++i) {
            labelParents[i] = toIntArray((Collection)labelParentsTmp.get(i));
            labelChildren[i] = toIntArray((Collection)labelChildrenTmp.get(i));
        }
        
        // These are not needed anymore, don't waste memory
        labelsTmp = null;
        labelParentsTmp = null;
        labelChildrenTmp = null;
        
        // Prepare for later
        labelState = new int[labels.length];
        sortLabels();
        computeDominators();
        printLabelsAndDominators();
        TimeUtil.stop("collect_nodes_time");
    }
    
    private void recPropagateUnsat(int x) {
        if (labelState[x] == UNSAT) return;
        //System.err.println("unsat " + x);
        labelState[x] = UNSAT;
        for (int i = 0; i < labelChildren[x].length; ++i) {
            int j;
            int c = labelChildren[x][i];
            for (j = 0; j < labelParents[c].length && labelState[labelParents[c][j]] == UNSAT; ++j);
            if (j == labelParents[c].length)
                recPropagateUnsat(c);
        }
    }
    
    private void propagateUnsat(int x) {
        recPropagateUnsat(x);
        computeDominators();
    }
    
    private void propagateSat(int x) {
        if (labelState[x] == SAT) return;
        //System.err.println("sat " + x);
        labelState[x] = SAT;
        propagateSat(dominator[x]);
    }
    
    // Ignores UNSAT nodes
    private void computeDominators() {
        int j;
        TimeUtil.start("compute_dominators_time");
        // Initialize the dominator tree
        dominator = new int[labels.length];
        for (int i = 0; i < labels.length; ++i) {
            for (j = 0; j < labelParents[i].length && labelState[labelParents[i][j]] == UNSAT; ++j);
            if (j == labelParents[i].length)
                dominator[i] = i;
            else 
                dominator[i] = labelParents[i][j];
        }
        // Process nodes in a topological order
        for (int i = 0; i < labels.length; ++i) {
            int l = labelsSort[i];
            if (labelState[l] == UNSAT) continue;
            for (j = 0; j < labelParents[l].length; ++j) {
                int d = labelParents[l][j];
                if (labelState[d] == UNSAT) continue;
                while (dominator[l] != d) {
                    if (labelsInvSort[dominator[l]] < labelsInvSort[d])
                        d = dominator[d];
                    else
                        dominator[l] = dominator[dominator[l]];
                }
            }
        }
        // Set the down links in the dominators tree
        dominated = new int[labels.length][];
        dominatedCnt = new int[labels.length];
        for (int i = 0; i < labels.length; ++i)
            ++dominatedCnt[dominator[i]];
        for (int i = 0; i < labels.length; ++i) {
            dominated[i] = new int[dominatedCnt[i]];
            dominatedCnt[i] = 0;
        }
        for (int i = 0; i < labels.length; ++i)
            dominated[dominator[i]][dominatedCnt[dominator[i]]++] = i;
        TimeUtil.stop("compute_dominators_time");
    }
    
    private void printLabelsAndDominators() {
        try {
            PrintWriter pw = new PrintWriter(new FileOutputStream("tmp.dot"));
            pw.println("digraph x {");
            for (int i = 0; i < labels.length; ++i) {
                for (int j = 0; j < labelChildren[i].length; ++j)
                    pw.println("" + i + "->" + labelChildren[i][j]);
                pw.println("" + i + "->" + dominator[i] + "[color=blue]");
            }
            pw.println("}");
            pw.flush();
        } catch (IOException e) {
            System.err.println("Can't dump the graph to tmp.dot!");
        }
    }

    private String getLabel(Node n) {
        if (n instanceof CodeNode) {
            CodeNode cn = (CodeNode) n;
            String label = assertLabel(cn.getCode());
            return label;
        }
        return null;
    }
    
    private int getParentsCnt(NaryExpr e) {
        Integer i = (Integer)parentsCnt.get(e);
        if (i == null) return 0;
        return i.intValue();
    }
    
    private void incParentsCnt(NaryExpr e) {
        parentsCnt.put(e, new Integer(getParentsCnt(e) + 1));
    }
    
    private void countParents(NaryExpr e) {
        if (seenExprs.contains(e)) return;
        seenExprs.add(e);
        for (int i = 0; i < e.exprs.size(); ++i) {
            Expr o = e.exprs.elementAt(i);
            if (!isOrAnd(o)) continue;
            NaryExpr c = (NaryExpr)o;
            incParentsCnt(c);
            countParents(c);
        }
    }
    
    /**
     * Gets (an approximation of) the print size of a tree.
     */
    private int getSize(NaryExpr tree) {
        Integer s = (Integer)treeToSizeCache.get(tree);
        if (s != null) return s.intValue();
        int sz = 1;
        for (int i = 0; i < tree.exprs.size(); ++i) {
            Expr e = tree.exprs.elementAt(i);
            if (isOrAnd(e)) 
                sz += getSize((NaryExpr)e);
            else 
                ++sz;
        }
        treeToSizeCache.put(tree, new Integer(sz));
        return sz;
    }
    
    // Returns whether |e| is a |NaryExpr| with BOOLAND or BOOLOR.
    private static boolean isOrAnd(Expr e) {
        if (e == null || !(e instanceof NaryExpr)) return false;
        int op = ((NaryExpr)e).op;
        return op == TagConstants.BOOLAND || op == TagConstants.BOOLOR;
    }
    
    /**
     * Transforms a DAG into an (almost-)tree by eliminating sharing of
     * big structures, where `big' means that the printed version would be
     * bigger with sharing than without.
     * 
     * This function is memoized using the maps dagToVarCache and
     * varToTreeCache.
     * 
     * This function adds the variables used in plucking to the 
     * pluckedVars set.
     * 
     * It assumes that dagParentsCnt maps |e| and all NaryExpr-essions
     * reachable from |e| to the number of parents they have.
     * 
     * pre: isOrAnd(dag)
     * 
     * TODO: review and maybe rewrite
     */
    private NaryExpr recDagToTree(NaryExpr dag) {
        Assert.notFalse(isOrAnd(dag));
        VariableAccess v = (VariableAccess)dagToVarCache.get(dag);
        if (v != null) return (NaryExpr)varToTreeCache.get(v);
        
        ExprVec ncv = ExprVec.make(); // new children vector
        for (int i = 0; i < dag.exprs.size(); ++i) {
            Expr e = dag.exprs.elementAt(i);
            if (!isOrAnd(e)) {
                ncv.addElement(e);
                continue;
            }
            NaryExpr c = (NaryExpr)e; // child
            NaryExpr nc = recDagToTree(c); // new (plucked) child
            int ncsz = getSize(nc); // print size of nc
            int ncp = getParentsCnt(c); // parents count
            if (ncsz * (ncp - 1) <= ncp + THRESHOLD) {
                ncv.addElement(nc);
                continue;
            }
            v = (VariableAccess)dagToVarCache.get(c);
            ncv.addElement(v);
        }
        
        // put the result in cache so we don't work twice
        NaryExpr tree = NaryExpr.make(dag.sloc, dag.eloc, dag.op, dag.methodName, ncv);
        v = GC.makeVar(NAME + varNameCnt++);
        dagToVarCache.put(dag, v);
        varToTreeCache.put(v, tree);
        
        return tree;
    }
    
    private void recGetUsedVars(NaryExpr e) {
        if (seenExprs.contains(e)) return;
        seenExprs.add(e);
        for (int i = 0; i < e.exprs.size(); ++i) {
            Expr c = e.exprs.elementAt(i);
            if (isOrAnd(c)) recGetUsedVars((NaryExpr)c);
            else if (c instanceof VariableAccess) {
                NaryExpr tree = (NaryExpr)varToTreeCache.get(c);
                if (tree != null) {
                    usedVariables.add(c);
                    recGetUsedVars(tree);
                }
            }
        }
    }
    
    private void getUsedVars(NaryExpr e) {
        seenExprs = new HashSet();
        usedVariables = new HashSet();
        recGetUsedVars(e);
    }
    
    private NaryExpr dagToTree(NaryExpr dag) {
        TimeUtil.start("dag_to_tree_time");
        // count the parents for each node reachable from dag
        parentsCnt = new HashMap();
        seenExprs = new HashSet(); countParents(dag);
        
        // do the plucking
        NaryExpr ne = recDagToTree(dag);
        
        // get the used variables and add their definitions
        getUsedVars(ne);
        ExprVec exprs = ExprVec.make();
        Iterator it = usedVariables.iterator();
        while (it.hasNext()) {
            VariableAccess v = (VariableAccess)it.next();
            Expr def = (Expr)varToTreeCache.get(v);
            //exprs.addElement(GC.equiv(v, def));
            // implication should be enough because the formulas are monotone
            exprs.addElement(GC.implies(v, def)); 
        }
        exprs.addElement(ne);
        NaryExpr result = (NaryExpr)GC.and(exprs);
        TimeUtil.stop("dag_to_tree_time");
        return result;
    }

    // Compute the strongest postcondition (SP) of [n].
    private Expr spDfs(Node n) {
        Expr sp = (Expr) spOfNode.get(n);
        if (sp != null) return sp;

        // compute precondition for [n] as a disjunction of its preconditions
        Enumeration p = n.getParents().elements();
        Expr pre = null;
        while (p.hasMoreElements()) {
            Expr onePre = spDfs(p.nextElement());
            if (pre == null) pre = onePre;
            else pre = GC.or(pre, onePre);
        }

        // compute the sp
        Expr post = n.computeSp(pre);
        
        // cache and store if interesting
        spOfNode.put(n, post);
        int idx = nodeToLabelIdx(n);
        if (idx != -1) preconditions[idx] = pre;
        return post;
    }
    
    private boolean hasUnsat(boolean[] set, boolean d) {
        ExprVec conjuncts = ExprVec.make();
        boolean[] exclude = new boolean[labels.length];
        for (int i = 0; i < labels.length; ++i) if (set[i]) {
            int cnt = 0, x = 0;
            for (int j = 0; j < labelParents[i].length; ++j) {
                int p = labelParents[i][j];
                if (labelState[p] == UNSAT) continue;
                ++cnt; x = p;
            }
            if (cnt == 1) exclude[x] = true;
        }
        for (int i = 0; i < labels.length; ++i) if (!exclude[i] && set[i])
            conjuncts.addElement(preconditions[i]);
        Expr query = GC.and(conjuncts);
        if (isOrAnd(query)) query = dagToTree((NaryExpr)query);
        if (Util.size(query, VC_LIMIT) == -1) {
            vcTooBig |= !d;
            return d;
        } else {
            /*
            System.err.print("query");
            for (int i = 0; i < labels.length; ++i) if (set[i])
                System.err.print(" " + i);
            System.err.println();
            */
            return Simplifier.isFalse(query);
        }
    }
    
    private boolean searchUnsat(boolean[] set, boolean hu) {
        int i, j;
        
        // eliminate known
        int cnt = 0, x = 0;
        for (i = 0; i < labels.length; ++i) {
            set[i] &= labelState[i] == UNKNOWN;
            if (set[i]) {
                ++cnt;
                x = i;
            }
        }
        if (cnt == 0) return false;
        
        // do the query
        if ((hu && cnt > 1) || hasUnsat(set, cnt > 1)) {
            if (cnt == 1) propagateUnsat(x);
            else {
                // split and recurse
                boolean[] s1 = new boolean[set.length];
                boolean[] s2 = set;
                for (i = j = 0; j < cnt / 2; ++i) if (set[labelsSort[i]]) {
                    ++j;
                    s1[labelsSort[i]] = true;
                    s2[labelsSort[i]] = false;
                }
                searchUnsat(s2, !searchUnsat(s1, false));
            }
            return true;
        } else {
            for (i = 0; i < labels.length; ++i) if (set[i]) 
                propagateSat(i);
            return false;
        }
    }
    
    private void reportAllLabels(Node n) {
        if (graphNodes.contains(n)) return;
        graphNodes.add(n);
        String label = getLabel(n);
        if (label != null) {
            NodeAndLabel nl = new NodeAndLabel(n, label);
            if (nl.loc != Location.NULL)
                reportUnreachable(nl);
        }
        Enumeration children = n.getChildren().elements();
        while (children.hasMoreElements())
            reportAllLabels(children.nextElement());
    }
    
    private void checkReachability() {
        int i, j;
        
        // mark nodes that are too big to analyze
        for (i = 0; i < labels.length; ++i) {
            if (isOrAnd(preconditions[i])) 
                preconditions[i] = dagToTree((NaryExpr)preconditions[i]);
            if (Util.size(preconditions[i], VC_LIMIT) == -1) { 
                labelState[i] = TOOBIG;
                vcTooBig = true;
            }
        }
        
        while (true) {
            // find the unknown node that has most dominators
            int[] mostDominators = new int[0];
            for (i = 0; i < labels.length; ++i) if (labelState[i] == UNKNOWN) {
                ArrayList/*<Integer>*/ dom = new ArrayList();
                j = i;
                dom.add(new Integer(j));
                do {
                    j = dominator[j];
                    if (labelState[j] == UNKNOWN) dom.add(new Integer(j));
                } while (j != 0);
                if (dom.size() > mostDominators.length)
                    mostDominators = toIntArray(dom);
            }
            
            // say you are done if no such node
            if (mostDominators.length == 0) return;
            
            // first see if that node is SAT
            if (!Simplifier.isFalse(preconditions[mostDominators[0]])) {
                propagateSat(mostDominators[0]);
                continue;
            }
            
            // do binary search to see where UNSAT starts
            int left = 0, right = mostDominators.length;
            while (left + 1 != right) {
                int middle = (left + right) / 2; // no overflow danger :)
                if (Simplifier.isFalse(preconditions[mostDominators[middle]]))
                    left = middle;
                else
                    right = middle;
            }
            propagateUnsat(mostDominators[left]);
            if (right < mostDominators.length)
                propagateSat(mostDominators[right]);
        }
    }

    private void internalAnalyze(GuardedCmd gc) {
        vcTooBig = false;
        
        // initialize the caches used by the DAG-to-(almost-)tree translation
        dagToVarCache = new HashMap();
        varToTreeCache = new HashMap();
        treeToSizeCache = new HashMap();
        
        // construct the graph, assumming that gc doesn't contain VARIN
        GCtoCFDBuilder builder = new GCtoCFDBuilder();
        graph = builder.constructGraph(gc, null);
        if (graph.isEmpty()) return; // nothing to check

        // DBG
        //System.out.println("Constructed graph: " + graph);
        //graph.printStats();
         
        // do DFS
        collectNodes();
        spOfNode = new HashMap();
        Node init = graph.getInitNode();
        preconditions[0] = GC.truelit;
        spOfNode.put(init, preconditions[0]);
        Iterator nodes = graphNodes.iterator();
        while (nodes.hasNext()) spDfs((Node)nodes.next());
        
        // check the interesting (labeled) nodes

        // multiple queries approach
        /*boolean[] all = new boolean[labels.length];
        for (int i = 0; i < labels.length; ++i) all[i] = true;
        searchUnsat(all, false);*/
        // single queries approach
        checkReachability();
        
        // report errors
        for (int i = 0; i < labels.length; ++i) if (labelState[i] == UNSAT) {
            int j;
            for (j = 0; j < labelParents[i].length && labelState[labelParents[i][j]] == UNSAT; ++j);
            if (j < labelParents[i].length) reportUnreachable(labels[i]);
        }
        
        // get really unreachable code
        // NOTE: I would rather have the CFD builder put an assume false and
        // not deal with this. It would be just as fast and nicer.
        List/*<CFD>*/ unreachable = builder.getUnreachable();
        Iterator uit = unreachable.iterator();
        while (uit.hasNext()) {
            CFD u = (CFD)uit.next();
            init = u.getInitNode();
            if (init != null) reportAllLabels(init);
        }
        
        if (vcTooBig) 
            ErrorSet.caution("The reachability analysis is incomplete because some VCs are too big.");
        
        //TimeUtil.total("compute_dominators_time");
        //TimeUtil.total("prover_time");
        //TimeUtil.total("dag_to_tree_time");
        //TimeUtil.total("collect_nodes_time");
    }
    
    private static final String POST_WARN =
        "The assumptions are inconsistent, the method might not terminate, or the analysis is imprecise.";
    private static final String CODE_WARN = "Code not checked.";
    private static final String ASSERT_WARN = "Assertion not checked."; 

    /**
     * Reports errors for unchecked code, unchecked assertions, and unchecked
     * preconditions. The location is not reported for preconditions.
     * We do not report @unreachable as being unreached. We skip
     * labels with no location information.
     */
    private void reportUnreachable(NodeAndLabel nl) {
        String suffix = " (" + nl.ld.getMsgShort() + ")";
        Assert.notFalse(nl.loc != Location.NULL);
        
        switch (nl.ld.getMsgTag()) {
        case TagConstants.CHKCODEREACHABILITY:
            return; // the user wants this to be unreachable so don't report
        case TagConstants.CHKPOSTCONDITION:
        case TagConstants.CHKUNEXPECTEDEXCEPTION:
        case TagConstants.CHKUNEXPECTEDEXCEPTION2:
            ErrorSet.warning(POST_WARN + suffix);
            break;
        case TagConstants.CHKASSERT:
        case TagConstants.CHKEXPRDEFINEDNESS:
        case TagConstants.CHKEXPRDEFNORMPOST:
        case TagConstants.CHKEXPRDEFEXCEPOST:
            ErrorSet.warning(nl.loc, ASSERT_WARN + suffix);
            break;
        default:
            // we assume it's code (for now)
            ErrorSet.warning(nl.loc, CODE_WARN + suffix);
        }
    }

    // pre: gc does not contain VARIN commands
    public static void analyze(GuardedCmd gc) {
        new ReachabilityAnalysis().internalAnalyze(gc);
    }
}
