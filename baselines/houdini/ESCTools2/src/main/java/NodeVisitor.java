package escjava.dfa.cfd;

/**
 * A stub for a search through a graph, a variant of a visitor patter for a graph.
 * @author mikolas
 *
 */
public abstract class  NodeVisitor {
    /*@ non_null @*/ NodeList visitedNodes;

    protected void resetSearch() {
        visitedNodes = new NodeList();
    }


    protected void visitChildren(/*@ non_null @*/Node n) {
        for (NodeList.Enumeration e = n.getChildren().elements(); e.hasMoreElements(); ) {
            Node child = e.nextElement();
            if (!visitedNodes.member(child)) {
                visitedNodes.addNode(child);
                child.accept(this);
            }
        }
    }

    public void visitCouplingNode(/*@ non_null @*/ CouplingNode n) {
        this.visitChildren(n);
    }

    public void visitCodeNode(/*@ non_null @*/ CodeNode n) {
        this.visitChildren(n);
    }

    public void visitExceptionNode(/*@ non_null @*/ ExceptionNode n) {
        this.visitChildren(n);
    }


}
