/*  Copyright 2004, David R. Cok 
    Originally generated as part of a GUI interface to the
    Esc/Java2 application.
*/

package escjava.gui;

import javax.swing.*;
import javax.swing.tree.*;
import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import javax.swing.event.*;
import java.util.Vector;
import java.util.Enumeration;
import java.util.ArrayList;
import java.util.Iterator;
import java.io.PrintStream;
import java.io.ByteArrayOutputStream;
import java.io.FileReader;
import javafe.ast.CompilationUnit;
import javafe.ast.Expr;
import javafe.ast.TypeDecl;
import javafe.ast.TypeDeclElem;
import javafe.ast.RoutineDecl;
import javafe.ast.MethodDecl;
import javafe.ast.Identifier;
import javafe.ast.Util;
import javafe.tc.Types;
import javafe.tc.OutsideEnv;
import javafe.util.Set;
import javafe.util.ErrorSet;
import javafe.util.Assert;
import javafe.util.FatalError;
import javafe.util.Location;
import javafe.genericfile.GenericFile;
import javafe.InputEntry;
import javafe.FileInputEntry;

import escjava.ast.LabelExpr;
import escjava.ast.GuardedCmd;
import escjava.backpred.FindContributors;
import escjava.translate.InitialState;
import escjava.translate.Targets;
import escjava.translate.VcToString;
import escjava.tc.TypeSig;
import escjava.tc.TypeCheck;
import escjava.translate.Ejp;
import escjava.translate.GC;
import escjava.translate.GCSanity;
import escjava.translate.LabelInfoToString;
import escjava.translate.NoWarn;
import escjava.translate.Translate;
import escjava.translate.UniqName;
import escjava.sp.DSA;
import escjava.sp.SPVC;

import escjava.Status;

import junitutils.Utils;


public class GUI extends escjava.Main {

    static public class Stop extends RuntimeException {}
    static public final Stop STOP = new Stop();

    volatile public boolean stop = false;
    private void stopCheck(boolean thr) { 
	if (!stop) return;
	//System.out.println("STOPPED");  
	if (thr) throw STOP; 
    }

    static GUI gui;
    static EscFrame escframe;
    static DefaultMutableTreeNode topNode = new DefaultMutableTreeNode("");
    static DefaultTreeModel treeModel = new DefaultTreeModel(topNode);
    EscOutputFrame currentOutputFrame;

    public static void main(String[] args) {
	//junitutils.Utils.disable = true;

	gui = new GUI();

	Thread t = new WindowThread();
	t.start();

	gui.run(args); // parses the options and builds the top level tree

	escframe = new EscFrame();

	Thread.currentThread().setPriority( Thread.currentThread().getPriority()-1); 
				// Leave the event queue at a higher priority

	processTasks();
    }

    /** Reinitializes everything; if the argument is null, the options are
	not reinitialized.
     */
    public void restart(String[] args) {
	clear(args != null);
	InputEntry.clear( options.inputEntries );
	run(args);
	escframe.init();
    }

    public /*@non_null*/javafe.Options makeOptions() {
	return new Options();
    }

    /** Extends escjava.Options just to set some defaults to more
	appropriate values for the GUI.
     */
    static public class Options extends escjava.Options {
	public Options() {
	    quiet = true;

            specspath = escjava.Main.jarlocation;

            try {
                processOption("-nowarn", new String[] { "Deadlock" }, 0);
                processOption("-source", new String[] { "1.4" }, 0);
            } catch (javafe.util.UsageError e) {
            } // FIXME
        }
    }

    /** This is overloaded because, instead of automatically running
	through all inputs, we want to build a gui and give the user
	control.
     */
    public void frontEndToolProcessing(ArrayList args) {
	preload();
	handleAllInputEntries();
/*
	loadAllFiles(args);
	postload(); // FIXME - waht about this ??
	preprocess();
	handleAllCUs();
	postprocess();
*/
    }

    public ArrayList extractChildren(DefaultMutableTreeNode d) {
	ArrayList list = new ArrayList();
/* Part of maintaining the tree between reloads, but that is disabled for now
	Enumeration e = d.children();
	while (e.hasMoreElements()) {
	    list.add(e.nextElement());
	}
*/
	d.removeAllChildren();
        return list;
    }

    public DefaultMutableTreeNode findIEMatch(InputEntry ie, ArrayList oc) {
	Iterator ii = oc.iterator();
	while (ii.hasNext()) {
	    DefaultMutableTreeNode d = (DefaultMutableTreeNode)ii.next();
	    IETreeValue v = (IETreeValue)d.getUserObject();
	    if (ie.match(v.ie)) {
		ii.remove();
		return d;
	    }
	}
	return null;
    }

    /** Builds the top-level tree, containing just the InputEntry nodes. */
    public void handleAllInputEntries() {
	ArrayList args = options().inputEntries;
	ArrayList oldchildren = extractChildren(topNode);
	Iterator i = args.iterator();
	while (i.hasNext()) {
	    InputEntry ie = (InputEntry)i.next();
	    ie = ie.resolve();
	    DefaultMutableTreeNode ienode = findIEMatch(ie,oldchildren);
	    if (ienode != null) {
		topNode.add(ienode);
	    } else {
		ienode = IETreeValue.makeNode(ie);
		topNode.add(ienode);
	    }
        }
	treeModel.reload();
    }

    public void buildCUTree(GFCUTreeValue cuvalue) {
	CompilationUnit cu = cuvalue.cu;
	DefaultMutableTreeNode cunode = cuvalue.holder;
	for (int j=0; j<cu.elems.size(); ++j) {
	    TypeDecl td = cu.elems.elementAt(j);
	    createTDNode(td, cunode);
	}
	treeModel.nodeChanged(cunode);
	if (escframe != null && escframe.guioptionPanel.settings.autoExpand) {
	    escframe.tree.expandPath(new TreePath(cunode.getPath()));
	    int n = cunode.getChildCount();
	    for (int nn=0; nn<n; ++nn) {
		DefaultMutableTreeNode d = (DefaultMutableTreeNode)cunode.getChildAt(nn);
		escframe.tree.expandPath(new TreePath(d.getPath()));
	    }
	}
    }

    public void createTDNode(TypeDecl td, DefaultMutableTreeNode cunode) {
	DefaultMutableTreeNode tdnode = TDTreeValue.makeNode(td);
	cunode.add(tdnode);
	for (int k=0; k<td.elems.size(); ++k) {
	    TypeDeclElem tde = td.elems.elementAt(k);
	    if (tde instanceof RoutineDecl) {
		DefaultMutableTreeNode tdenode = RDTreeValue.makeNode((RoutineDecl)tde);
		tdnode.add(tdenode);
	    } else if (tde instanceof TypeDecl) {
		createTDNode( (TypeDecl)tde, cunode);
	    }
	}
    }

    public void doAll(int action) {
        if (escframe.guioptionPanel.settings.breadthFirst) {
	    for (int a=0; a<=action; ++a) {
		Enumeration children = topNode.children();
		while (children.hasMoreElements()) {
		    Object o = children.nextElement();
		    DefaultMutableTreeNode tn = (DefaultMutableTreeNode)o;
		    EscTreeValue cun = (EscTreeValue)tn.getUserObject();
		    cun.processHelper(a);
		}
	    }
	} else {
	    Enumeration children = topNode.children();
	    while (children.hasMoreElements()) {
		Object o = children.nextElement();
		DefaultMutableTreeNode tn = (DefaultMutableTreeNode)o;
		EscTreeValue cun = (EscTreeValue)tn.getUserObject();
		cun.process(action);
	    }
	}
    }

    // This applies the given action to the given type, doing the
    // parents as well if needed, but not recursively doing all of
    // the children.
    //@ requires action == TYPECHECK || action == CHECK;
    public int processTypeDecl(TDTreeValue tdn, int action) {
	TypeDecl td = tdn.td;
	tdn.scope = null;
	tdn.initState = null;
	tdn.sig = null;
	tdn.outputText = null;

	int status = -1;

	try {

	// Duplicates code between here and Main - refactor // FIXME
	int errorCount = ErrorSet.errors;
	int cautionCount = ErrorSet.cautions;
	javafe.tc.TypeSig sig = TypeCheck.inst.getSig(td); 
	sig.typecheck();
	NoWarn.typecheckRegisteredNowarns();
	tdn.sig = sig;
	if (stages >= 2 && ErrorSet.errors == errorCount) {
	    tdn.scope = new FindContributors(sig);
	    VcToString.resetTypeSpecific();
	}
        if (stages >= 3 && ErrorSet.errors == errorCount) {

            LabelInfoToString.reset();
            tdn.initState = new InitialState(tdn.scope);
            LabelInfoToString.mark();
	}

        if (ErrorSet.errors > errorCount) {
                status = Status.TYPECHECKED_ERROR;
        } else if (ErrorSet.cautions > cautionCount) {
                status = Status.TYPECHECKED_CAUTION;
        } else {
                status = Status.TYPECHECKED_OK;  
        }

	} catch (Stop e) {
	    status = Status.NOTPROCESSED;
	    throw e;
	} catch (FatalError e) {
	    status = Status.TYPECHECKED_ERROR;
	} catch (Throwable t) {
	    System.out.println("Exception thrown while handling a type declaration: " + t);
	    t.printStackTrace(System.out);
	}
	return status;
    }

    // This applies the given action to the given routine, doing the
    // parents as well if needed.
    // requires action == PARSE;
    public int processRoutineDecl(RDTreeValue rdn, int action) {
	RoutineDecl r = rdn.rd;
	TDTreeValue tdn = (TDTreeValue)rdn.getParent();

	int status = Status.ILLEGAL;
	ErrorSet.mark();
	try {

	if (r.body == null) {
	    status = Status.STATICCHECKED_PASSEDIMMED; // FIXME what about cautions
	} else {
	    Translate.globallyTurnOffChecks(gctranslator.inlineCheckDepth > 0);
	    LabelInfoToString.resetToMark();
	    GuardedCmd gc = computeBody(r, tdn.initState);
	    UniqName.resetUnique();
	    if (gc == null) {
		status = Status.STATICCHECKED_OK; // passed without checking
	    } else {
		Set directTargets = Targets.direct(gc);
		GCSanity.check(gc);
		// FIXME if (stages<4) ???
		gc = DSA.dsa(gc);
		// FIXME if (stages<5) ???
		Expr vcBody;
		if (options().spvc) {
		    vcBody = SPVC.compute(gc);
		} else {
		    vcBody = Ejp.compute(gc, GC.truelit, GC.truelit);
		}
		Expr vc = GC.implies(tdn.initState.getInitialState(), vcBody);
		String label = "vc." + tdn.sig.toString() + ".";
		if (r instanceof MethodDecl)
		    label += ((MethodDecl)r).id;
		else
		    label += "<constructor>";
		label += "." + UniqName.locToSuffix(r.getStartLoc());
		vc = LabelExpr.make(r.getStartLoc(), r.getEndLoc(),
		    false, Identifier.intern(label), vc);

		// Check for VC too big:
		int usize = Util.size(vc, options().vclimit);
		if (usize == -1) {
		    ErrorSet.caution("Unable to check "
				     + TypeCheck.inst.getName(r)
				     + " of type " + TypeSig.getSig(r.parent)
				     + " because its VC is too large");
		    status = Status.STATICCHECKED_TIMEOUT;
		} else {
		    stopCheck(true);

		    // FIXME if (stages<6) ???

		    status = Status.STATICCHECKED_ERROR;

		    GUI.gui.escframe.showGuiLight(1);
		    status = doProving(vc,r,directTargets,tdn.scope);
		    //System.out.println("DOPROVING " + status);
		    GUI.gui.stopCheck(true);
		}
	    }
	}

	} catch (Stop t) {
	    status = Status.NOTPROCESSED;
	    throw t;
	} catch (escjava.prover.SubProcess.Died t) {
	    ErrorSet.error("Prover died");
	    status = Status.STATICCHECKED_ERROR;
	} catch (Throwable t) {
	    status = Status.STATICCHECKED_ERROR;
	    System.out.println("An exception was thrown while processing a routine declaration: " + t);
	    t.printStackTrace(System.out);
	} finally {
	    GUI.gui.escframe.showGuiLight(2);
	    if (ErrorSet.errorsSinceMark()) 
	        status = Status.STATICCHECKED_ERROR;
	    else if (ErrorSet.cautionsSinceMark() && status == Status.STATICCHECKED_OK)
		status = Status.STATICCHECKED_CAUTION;
	}
	return status;
    }

    static abstract class EscTreeValue {
	public int status = Status.NOTPROCESSED; 
	public boolean outOfDate = false;
	public void setOutOfDate() {
	    outOfDate = true;
	    setOutOfDate(holder);
	}
	static public void setOutOfDate(DefaultMutableTreeNode h) {
	    Enumeration e = h.children();
	    while (e.hasMoreElements()) {
		Object o = e.nextElement();
		if (!(o instanceof DefaultMutableTreeNode)) return;
		DefaultMutableTreeNode d = (DefaultMutableTreeNode)o;
		if (d.getUserObject() instanceof EscTreeValue) {
		    EscTreeValue v = (EscTreeValue)d.getUserObject();
		    v.setOutOfDate();
	        }
	    }
	}
	public /*@non_null*/String getStatusText() {
	    return type() + (outOfDate ? " (OUT OF DATE) " : "" )
		+ infoString() + ": " + Status.toString(status);
	}
	public /*@non_null*/String infoString() { return toString(); }
	public String outputText = null;
	public DefaultMutableTreeNode holder;
        public int action; // so this can double as a process task
	abstract public String type();
	public EscTreeValue getParent() {
	    if (holder == null) return null;
	    Object o = ((DefaultMutableTreeNode)holder.getParent()).getUserObject();
	    if (o instanceof EscTreeValue) return (EscTreeValue)o;
	    return null;
	}
	public void setStatus(int status, String outputText) {
	    this.status = status;
	    this.outputText = outputText;
	    if (holder != null) treeModel.nodeChanged(holder);
	    if (escframe != null && escframe.guioptionPanel.settings.autoScroll)
		escframe.tree.scrollPathToVisible(new TreePath(holder.getPath()));
	}
	public void showOutput(boolean showEmpty) {
	    if (!showEmpty && (outputText == null || outputText.length() == 0)) return;
	    if (showEmpty || (escframe != null && escframe.guioptionPanel.settings.autoShowErrors)) {
		if (GUI.gui.currentOutputFrame == null) 
		    GUI.gui.currentOutputFrame = new EscOutputFrame(combinedString(),outputText);
		else {
		    GUI.gui.currentOutputFrame.setText(outputText);
		    GUI.gui.currentOutputFrame.setTitle(combinedString());
		}
	    }
	}
	public /*@non_null*/String combinedString() { return toString(); }

	// Recursively processes the given action for everything below
	// this node in the tree
	public void process(int action) {
	    //System.out.println("PROCESS-" + type() + " " + this + " " + action + " " + Status.toString(status));
	    if (escframe.guioptionPanel.settings.breadthFirst) {
		for (int a=0; a<=action; ++a) {
		    processHelper(a);
		}
	    } else {
		processHelper(action);
	    }
	    //System.out.println("ENDPROCESS-" + type() + " " + this + " " + action + " " + Status.toString(status));
        }
	public void processHelper(int action) {
	    if (outOfDate) {
		outOfDate = false;
		status = Status.NOTPROCESSED;
	    }
	    processThis(action);
	    if (status == Status.PARSED_ERROR) return;
	    if (status == Status.TYPECHECKED_ERROR) return;
	    Enumeration children = holder.children();
	    while (children.hasMoreElements()) {
		DefaultMutableTreeNode n = (DefaultMutableTreeNode)children.nextElement();
		EscTreeValue v = (EscTreeValue)n.getUserObject();
		v.processHelper(action);
	    }
	}

	// Processes just this node, not the children, if there is anything to do
	public void processThis(int action) {
	    GUI.gui.stopCheck(true);
	    if (action == CLEAR) { clearCheck(); return; }
	    int actualAction = actualAction(action);
	    if (actualAction == -10) return;
	    if (actionComplete(status, actualAction)) return;

	    EscTreeValue ien = getParent();
	    if (ien != null) {
		if (!actionComplete(ien.status, actualAction)) 
						ien.processThis(actualAction);
		if (Status.isError(ien.status)) return;
	    }
	    escframe.label.setText(actionString(actualAction) + toString());
	    ByteArrayOutputStream ba = junitutils.Utils.setStreams();
	    try {
		status = processThisAction(actualAction);
	    } catch (Stop t) {
		status = Status.NOTPROCESSED;
		throw t;
	    } catch (Throwable t) {
		System.out.println("An exception was thrown while processing."
			+ Project.eol + t);
		t.printStackTrace(System.out);
	    } finally {
		junitutils.Utils.restoreStreams(true);
		String out = ba.toString();
		if (status == Status.NOTPROCESSED) out = "";
		setStatus(status,out);
		escframe.label.setText(" ");
		showOutput(false);
		if (ien != null) ien.propagateStatus(status);
	    }
	}
        abstract public int actualAction(int action);
	abstract public int processThisAction(int action);
	abstract public String getFilename();
	abstract public int getLine();

        public void clearCheck() {
		// Only TDTreeValues and RDTreeValues do anything
	}

        public void propagateStatus(int s) {
	    if ((Status.isOK(status) && !Status.isOK(s)) ||
		 (!Status.isError(status) && Status.isError(s))) {
		setStatus(Status.CHILDERROR,outputText);
		EscTreeValue v = getParent();
		if (v != null) v.propagateStatus(s);
	    }
	    if (status == Status.TYPECHECKED_WAITING) {
		Enumeration c = holder.children();
		int newstat = Status.PARSED_OK;
		while (c.hasMoreElements()) {
		    Object o = c.nextElement();
		    DefaultMutableTreeNode d = (DefaultMutableTreeNode)o;
		    o = d.getUserObject();
		    int ss = ((EscTreeValue)o).status;
		    if (ss < newstat) newstat = ss;
		}
		if (status != newstat) setStatus(newstat,outputText);
	    }
	}
    }

    static class IETreeValue extends EscTreeValue {
	static public DefaultMutableTreeNode makeNode(InputEntry ie) {
	    EscTreeValue v = new IETreeValue(ie);
	    DefaultMutableTreeNode n = new DefaultMutableTreeNode(v);
	    v.holder = n;
	    v.status = Status.NOTPROCESSED;
	    return n;
	}
	public IETreeValue(InputEntry ie) {
	    this.ie = ie;
	}
	public InputEntry ie;
	public String type() { return "Input Entry (" + ie.type() + ") "; }
	public /*@non_null*/String toString() {
	    return ie.name;
	}
	public String getFilename() {
	    if (ie instanceof FileInputEntry)
		return ie.name;
	    else
		return null;
	}
	public int getLine() { return 1; }
	public int actualAction(int action) {
	    return RESOLVE;
	}

        public int processThisAction(int action) {
	    ArrayList a = null;
	    ErrorSet.mark();
	    int s = Status.RESOLVED_ERROR;

	    //ByteArrayOutputStream ba = Utils.setStreams();
	    try {
		a = gui.resolveInputEntry(ie).contents;
		s = Status.RESOLVED_OK;
	    } catch (Exception e) {
		System.out.println("Failed to resolve " + ie.name + ": " + e);
		s = Status.RESOLVED_ERROR;
	    } finally {
		Utils.restoreStreams(true);
	    }
	    if (ErrorSet.errorsSinceMark()) s = Status.RESOLVED_ERROR;
	    else if (ErrorSet.cautionsSinceMark()) s = Status.RESOLVED_CAUTION;

	    if (holder.isLeaf() && a != null && a.size() != 0) {
		// The contents have not been added to the node tree
		Iterator i = a.iterator();
		while (i.hasNext()) {
		    GenericFile gf = (GenericFile)i.next();
		    DefaultMutableTreeNode d = GFCUTreeValue.makeNode(gf,Status.NOTPROCESSED);
		    holder.add(d);
		}
		if (escframe != null && escframe.guioptionPanel.settings.autoExpand) 
			escframe.tree.expandPath(new TreePath(holder.getPath()));
	    }
	    return s;
        }
    }

    static class GFCUTreeValue extends EscTreeValue {
	static public DefaultMutableTreeNode makeNode(GenericFile gf, int s) {
	    EscTreeValue v = new GFCUTreeValue(gf);
	    DefaultMutableTreeNode n = new DefaultMutableTreeNode(v);
	    v.holder = n;
	    v.status = s;
	    return n;
	}
	public GFCUTreeValue(GenericFile gf) {
	    this.gf = gf;
	    this.cu = null;
	}
	public GenericFile gf;
	public CompilationUnit cu;
	public String type() { return "Compilation Unit "; }
	public /*@non_null*/String toString() {
	    return gf.getHumanName();
	}
	public /*@non_null*/String infoString() {
	    if (cu == null) return toString();
	    if (!(cu instanceof escjava.RefinementSequence)) return toString();
	    ArrayList a = ((escjava.RefinementSequence)cu).refinements();
	    if (a.size() <= 1) return toString();
	    Iterator i = a.iterator();
	    StringBuffer s = new StringBuffer(((CompilationUnit)i.next()).sourceFile().getHumanName());
	    while (i.hasNext()) {
		String ss = ((CompilationUnit)i.next()).sourceFile().getHumanName();
		int n = ss.lastIndexOf('.');
		s.append("," + ss.substring(n));
	    }
	    return s.toString(); 
	}
	public String getFilename() { return gf.getHumanName(); }
	public int getLine() { return 1; }

        public int actualAction(int action) {
	    if (action == RESOLVE) return -10;
	    return PARSE;
	}

        public int processThisAction(int action) {
		// parse the GenericFile
	    ErrorSet.mark();
	    int s = Status.PARSED_ERROR; 
	    CompilationUnit cu = null;
	    try {
		cu = OutsideEnv.addSource(gf);
		s = Status.PARSED_OK;
	    } catch (Stop t) {
		s = Status.NOTPROCESSED;
		ErrorSet.mark();
		throw t;
	    } catch (Throwable e) {
		System.out.println("Parsing failed for " + gf.getHumanName()
			+ ": " + e);
		e.printStackTrace(System.out);
		s = Status.PARSED_ERROR;
	    } finally {
		if (ErrorSet.errorsSinceMark()) s = Status.PARSED_ERROR;
		else if (ErrorSet.cautionsSinceMark()) s = Status.PARSED_CAUTION;
		if (cu != null)  {

		    this.cu = cu;

		    if (holder.isLeaf()) {
			// FIXME - if there are already children are they valid ???
			// The contents have not been added to the node tree
			gui.buildCUTree(this);
		    }
		    // FIXME use TYPECHECKED_WAITING???
		} else {
		    s = Status.PARSED_ERROR;
		}
	    }
	    return s;
        }
    }

    static class TDTreeValue extends EscTreeValue {
	static public DefaultMutableTreeNode makeNode(TypeDecl td) {
	    EscTreeValue v = new TDTreeValue(td);
	    DefaultMutableTreeNode n = new DefaultMutableTreeNode(v);
	    v.holder = n;
	    v.status = Status.NOTPROCESSED;
	    return n;
	}
	public TDTreeValue(TypeDecl td) {
	    this.td = td;
	}
	public TypeDecl td;
	public javafe.tc.TypeSig sig;
	public InitialState initState;
	public FindContributors scope;
	public String type() { return "Type Declaration "; }
	public /*@non_null*/String toString() {
	    return TypeSig.getSig(td).toString();
	}
        public void clearCheck() {
	    if (status == Status.CHILDERROR) setStatus(Status.NOTPROCESSED,null);
	}
	public int actualAction(int action) {
	    if (action == CHECK) return TYPECHECK;
	    if (action != TYPECHECK) return -10;
	    return action;
	}
        public int processThisAction(int action) {
	    return gui.processTypeDecl(this,action);
        }
	public String getFilename() { return Location.toFileName(td.getStartLoc()); }
	public int getLine() { return Location.toLineNumber(td.getStartLoc()); }
    }

    static class RDTreeValue extends EscTreeValue {
	static public DefaultMutableTreeNode makeNode(RoutineDecl rd) {
	    EscTreeValue v = new RDTreeValue(rd);
	    DefaultMutableTreeNode n = new DefaultMutableTreeNode(v);
	    v.holder = n;
	    v.status = Status.NOTPROCESSED;
	    return n;
	}
	String sig;
	public RDTreeValue(RoutineDecl rd) {
	    this.rd = rd;
	    sig = rd.id().toString() + Types.printName(rd.argTypes());
	}
	public RoutineDecl rd;
	public String type() { return "Routine Declaration "; }
	public /*@non_null*/String toString() {
	    return sig;
	    //return rd.id().toString();
	}
	public String getFilename() { return Location.toFileName(rd.getStartLoc()); }
	public int getLine() { return Location.toLineNumber(rd.getStartLoc()); }
        public void clearCheck() {
	    if (Status.parsingComplete(status)) setStatus(Status.NOTPROCESSED,null);
	}
	public /*@non_null*/String combinedString() { return getParent() + "." + toString(); }
	public int actualAction(int action) {
	    if (action != CHECK) return -10;
	    return action;
	}
        public int processThisAction(int action) {
	    return gui.processRoutineDecl(this,action);
        }
    }

    static final int CHECK = 3;
    static final int TYPECHECK = 2;
    static final int PARSE = 1;
    static final int RESOLVE = 0;
    static final int CLEAR = -1;
    static final int RELOAD = -2;
    public static String actionString(int a) {
	if (a == CHECK) return "Checking ";
	if (a == TYPECHECK) return "Typechecking ";
	if (a == PARSE) return "Parsing ";
	if (a == RESOLVE) return "Resolving ";
	if (a == CLEAR) return "Clearing ";
	if (a == RELOAD) return "Reloading files ";
	return "Unknown Action ";
    }

    static final boolean actionComplete(int status, int action) {
	if (action == CHECK) {
	    return Status.checkComplete(status);
	} else if (action == PARSE) {
	    return Status.parsingComplete(status);
	} else if (action == TYPECHECK) {
	    return Status.typecheckComplete(status);
	} else  {
	    return Status.resolved(status);
	}
    }

    static TaskQueue processTasks = new TaskQueue();

    static void processTasks() {
	while (true) {
	    try {

	    GUI.gui.escframe.showGuiLight(0);
	    Object o = processTasks.getTask();
	    GUI.gui.escframe.showGuiLight(2);
	    if (o instanceof Integer) {
		int action = ((Integer)o).intValue();
		//System.out.println("PROCESSING EVERYTHING " + action);
		if (action == RELOAD) {
		    gui.restart(null);
		    //EscTreeValue.setOutOfDate(topNode);
		} else {
		    gui.doAll(action);
		}
	    } else if (o instanceof EscTreeValue) {
		EscTreeValue v = (EscTreeValue)o;
		//System.out.println("PROCESSING " + v.action + " " + o.getClass() + " " + o.toString());
		v.process(v.action);
	    } else if (o instanceof Stop) {
		GUI.gui.stop = false;
	    } else {
		System.out.println("UNKNOWN TASK: " + o);
	    }

	    } catch (Stop e) {
		    //System.out.println("CAUGHT STOP");
	    } catch (Throwable t) {
		JOptionPane.showMessageDialog(GUI.gui.escframe,
		    "An exception was thrown while processing: " + t
		    + Project.eol);
		t.printStackTrace(System.out);
	    }
	}
    }

}
	

/* Issues:

- IF a file in an RS has a parse problem (e.g. missing semicolon), no message
is reported 

- Problem with Java on XP

- Is there a problem with options being reset when a project is loaded?

- Need a button to reset all options to defaults

- Combine the files of a RS into one entry in the GUI

- Want skeletons for a whole missing file, a new file in ref sequence, a new method in a file
- Integrate other tools - JML, Daikon
- Check what happens if there is an error in the RS but not the given file
- Chekc the BLue for timeout/big VC
- What happens if typechecking depdns on new options after a clear.
- Check all the GUI/ESC options

- Want browse buttons for Classpath, Specpath, Input content
- Need checks that Simplify, classpath, spec patah are well-formed
- Highlighting a selection on scrolling in an editor does not work
- Need more ESC/Options
- Would like wild cards for directories, packages, recursive options
- Should dispose of editors before there are too many; perhaps limit the number
- Provide a menu of created editors

- Perhaps a button to set the NoWarn options to defaults

- On windows, the disable All button takes a long time to execute (should not
really rebuild the whole pane).

- Faster stopping of typechecking
- Guard against race conditions between checking and new/open projects

- Change the handling of duplicate inputs

- would like expand/collapse commands

- be able to generate skeleton specs

- Show loaded file info, including spec-only

- Show other internal info

- Write documentation

- Check that all tooltips are available
 
- Show project name in title
- Allow editing of order of items inthe tree?
- Show timem information
- Allow skipping of tree items
- Cleaing, repainting after an input change takese too long
- How to show refinement sequences when creasting an editor
- Display a warning when nothing is selected
- CHeck all the ... on menu items in Windows

- Refactor duplicate VC generation code 

- The auto scrolling does not always go to a predicatable position

- Provide a way to browse for a file to show in an editor

- Cache the documentation windows

- Test the combingin of the error ooutput from the methods of a class

- Allow multiple error windows (and cache them?)

- Mac won't draw lines in the tree

- Implement About
Accelerator keys
	New - N
	Save - S
	Open - O
	Editor - E
	Errors - R
	Check All - Shift K
	Check Selected - K
	Clear ALl - Shift L
	Clear Selected - L
	Expand ?
	Collaps ?
	Select All - A
	Clear Selections - Shift A  (need on menu also)
	Stop - Z

- Be able to filter error messages
- keep track of file times and allow updates; track dependencies

- auto skip items

- need ssome basic browser capabilty in the documentattion - can we launch a real brwoser?

- Add an GUI option to turn on/off the memory timer task; set the rate.

- Enable the Usage menu item, add content

- Figure out how to reliably set the user.dir from the application

- the pgc pdsa pvc options don't work (removing duplicated code will help)

- why does the memory usage keep rising when nothing is happening


*/
