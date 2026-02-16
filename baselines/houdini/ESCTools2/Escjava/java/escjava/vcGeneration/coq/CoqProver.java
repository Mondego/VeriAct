package escjava.vcGeneration.coq;

import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.LineNumberReader;
import java.io.Writer;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.Vector;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javafe.ast.Expr;
import escjava.Main;
import escjava.translate.GC;
import escjava.translate.InitialState;
import escjava.vcGeneration.ProverType;
import escjava.vcGeneration.TDisplay;
import escjava.vcGeneration.TMethodCall;
import escjava.vcGeneration.TNode;
import escjava.vcGeneration.TRoot;
import escjava.vcGeneration.TVisitor;
import escjava.vcGeneration.TypeInfo;
import escjava.vcGeneration.VariableInfo;
import escjava.vcGeneration.coq.visitor.simplifiers.TAndRemover;
import escjava.vcGeneration.coq.visitor.simplifiers.TNotRemover;
import escjava.vcGeneration.coq.visitor.simplifiers.CoqTProofSimplifier;
import escjava.vcGeneration.coq.visitor.simplifiers.TProofSplitter;
import escjava.vcGeneration.coq.visitor.simplifiers.TProofTyperVisitor;


/**
 * This class is an implementations of the interface {@link ProverType}.
 * In order to handle the Coq theorem prover.
 * @author J. Charles, based on the work of C. Hurlin and C.Pulley
 * @version 14/11/2005
 */
public class CoqProver extends ProverType {
    private static final String DEFAULT_PROOFSCRIPT = "startsc...\n";

	/** the prelude path: ie. defs.v */
    private static final String PRELUDE_PATH =  "defs.v"; 
    
    /** the proof obligations location: ie. coq_proofs */
    private static final String PROOF_PATH = "coq_proofs";

    
    /**
     * Pretty print a number, based on the size of the max number 
     * in the serie. ie. if the max is 123, the number 2 will be
     * printed 002.
     * @param n the number to pretty print
     * @param max the max number of the serie
     * @return the pretty version of <code>n</code>
     */
    public static String ppNumber(int n, int max) {
    	int oldmax = max;
    	int oldn = n;
    	int i = 0;
    	for (; max != 0; max = max /10) i++;
    	int j = 0;
    	max = oldmax;
    	
    	for (; n != 0; n = n /10) j++;
    	max = i - j;
    	String res = "";
    	for(int k = 0; k < max; k++) {
    		res += "0";
    	}
    	return res + oldn;
    }
    
    
    
	/**
	 * @return an instance of the class {@link TCoqVisitor}.
	 */
    public TVisitor visitor(Writer out) {
        //return new SimpleVisitor(out, this);
    	return new TCoqVisitor(out, this);
    }

    /**
     * Write in the directory coq_proofs/proofName the different
     * vernacular files corresponding to the proof obligations.
     * @param proofname the name we should give to the generated lemma.
     * @param declns the declarations of the forall.
     * @param vc the generated verification condition.
     * @return the Coq vernacular file representing the proof.
     */
    public void getProof(Writer output, String proofName, TNode term) {
    	String src = Main.options.userSourcePath;
    	
    	if(src == null) {
    		// moi je dis blurb!
    		// this fix is for use with eclipse.
    		// as a disclaimer with esc/java eclipse plugin 
    		// Main.options.userSourcePath should never be null
    		List l = Main.options.inputEntries;
    		Iterator iter = l.iterator();
    		while(iter.hasNext()) {
    			//we try to find an existing file
    			File f = new File(iter.next().toString());
    			if(f.exists()) {
    				File res = f.getParentFile();
            if (res != null) {
              if(res.getName().equals("src")) {
                res = res.getParentFile();
              }
              src = res.getAbsolutePath();
              break;
            }
    			}
    		}
    	}
    	TDisplay.info("I say: I have found this path... " +src + "!");
    	File coqProofDir = new File(src, PROOF_PATH);
    	if(!coqProofDir.exists()) {
    		coqProofDir.mkdir();
    	}
    	Prelude p = new Prelude(new File(coqProofDir, PRELUDE_PATH));
    	try {
			p.generate();
		} catch (IOException e) {
			e.printStackTrace();
		}
    	String className = proofName.substring(3, proofName.lastIndexOf("_"));
    	className = className.substring(0, className.lastIndexOf("_"));
    	if(className.charAt(className.length() - 1) == '_') {
    		// _constructor_ case
    		className = className.substring(0, className.lastIndexOf("_"));
    		className = className.substring(0, className.lastIndexOf("_"));
    	}
    	className = className.substring(0, className.lastIndexOf("_"));
    	String methodName = proofName.substring(4 + className.length());
    	int ind = methodName.lastIndexOf('_', methodName.lastIndexOf('_') - 1);
    	methodName = methodName.substring(0, ind) + "." + methodName.substring(ind +1);
    	System.out.println(className);
    	File coqCurrentClassDir = new File(coqProofDir, className);
    	coqCurrentClassDir.mkdir();
    	File coqCurrentProofDir = new File(coqCurrentClassDir, methodName);
    	coqCurrentProofDir.mkdir();
    	if (!(term instanceof TRoot)) {
    		TDisplay.err("For coq pretty printer, the node should be a root node!" + 
    				term.getClass());
    		return;
    	}
    	TRoot root = (TRoot) term;
    	
    	Iterator iter = root.sons.iterator();
    	int size = root.sons.size();
    	int count = 1;
    	while(iter.hasNext()) {
    		File proof = new File(coqCurrentProofDir, "goal" + ppNumber(count, size) + ".v");
    		term = (TNode) iter.next();
    		String script = getProofScript(proof);
    		writeProofObligation(proofName, proof, term, script);
        	count++;
    	}
    }

    
    /**
     * This method modify the proof obligation term given as a
     * parameter. It returns a list of proof terms corresponding
     * to the original one.
     * @param term the term to simplify and/or decompose
     * @return the list of term corresponding to the original term
     * @throws IOException
     */
	private TRoot simplifyProofObligation(TRoot term) {
        TProofTyperVisitor tptv = new TProofTyperVisitor();
        term.accept(tptv);
		CoqTProofSimplifier tps = new CoqTProofSimplifier();
    	term.accept(tps);
    	TNotRemover tfr = new TNotRemover();
    	term.accept(tfr);
    	TAndRemover tar = new TAndRemover();
    	term.accept(tar);
    	TProofSplitter ps = new TProofSplitter();
    	term.accept(ps);
		return term;
	}
    
    
    

    
    
    /**
     * Returns the proof script previously available in the proof
     * or a default proof script in the worst case.
     * @param proof the proof to parse in order to find the old
     * proof script
     * @return the old proof script or {@link #DEFAULT_PROOFSCRIPT}
     * @throws IOException if there is a problem reading the proof
     * file
     */
	private String getProofScript(File proof) {
		if(proof.exists()) {
			try {
				LineNumberReader lnr = new LineNumberReader(new FileReader(proof));
				String red;
				while((red = lnr.readLine()) != null) {
					if(red.startsWith("Proof with autosc")) {
						String res = "";
						while(((red = lnr.readLine()) != null)
								&& !red.startsWith("Qed.")) {
							res += red + "\n";
						}
						lnr.close();
						return res;
					}
				}
				lnr.close();
			}
			catch (IOException e) {
				e.printStackTrace();
			}
		}
		return DEFAULT_PROOFSCRIPT;
	}

	
	/**
	 * Write the proof obligation to the disc.
	 * @param proofName the name of the proof
	 * @param proof the file where to write the proof
	 * @param term the proof obligation to translate to Coq
	 * @param proofScript the proof script to put inside of the proof obligation
	 * @throws IOException if there is an error while writing the file
	 */
	private void writeProofObligation(String proofName, File proof, TNode term, String proofScript) {
		try {
			Writer out = new FileWriter(proof);
			out.write("Load \"coq_proofs" + File.separator + PRELUDE_PATH + "\".\n");
			generatePureMethodsDeclarations(out);
			out.write("Lemma " + proofName + " : \n");
			out.write("forall ");
			generateDeclarations(out, term);
			out.write(" ,\n");
			generateTerm(out, term);
			out.write(".\n");
			out.write("Proof with autosc.\n"+ proofScript + "Qed.\n");
			out.close();
		} 
		catch (IOException e) {
			
		}
	}

    
    /**
     * Generates the pure methods declarations
     * @param s The output for the declarations.
     */
    private void generatePureMethodsDeclarations(Writer s) throws IOException {
        //Let's go for the purity:
        Vector methNames = TMethodCall.methNames;
        Vector methDefs = TMethodCall.methDefs;
        for (int i = 0; i < methNames.size(); i++) {
            s.write("Variable "
                    + getVariableInfo(((VariableInfo) methNames.get(i)))
                    + " : ");
            TMethodCall tmc = (TMethodCall) (methDefs.get(i));
            Vector v = tmc.getArgType();
            for (int j = 0; j < v.size(); j++) {
                TypeInfo ti = ((TypeInfo) v.get(j));
                if (ti == null)
                    s.write("S -> ");
                else
                    s.write(((TypeInfo) v.get(j)).def + " -> ");
            }
            if (tmc.type == null) {
                s.write("S.\n");
            } else {
                s.write(tmc.type.def + ".\n");
            }
        }
    }

    
    /**
     * @param caller the type to translate to Coq.
     * @return a valid type name.
     */
    public String getTypeInfo(TypeInfo caller) {
        if (caller.def == null)
            coqRename(caller);
        return caller.def;
    }

    
    /**
     * Defines predefined Coq types, and some predefined variables
     * (ecReturn, ecThrow and XRES).
     * For now there is no String types should be extended in the future.
     */
    public void init() {
        // Predefined types
        TNode._Reference = TNode.addType("%Reference", "Reference");
        TNode._Time = TNode.addType("%Time", "Time");
        TNode._Type = TNode.addType("%Type", "Types");
        TNode._boolean = TNode.addType("boolean", "bool");
        TNode._char = TNode.addType("char", "char");
        TNode._DOUBLETYPE = TNode.addType("DOUBLETYPE", "double"); 
        TNode._double = TNode.addType("double", "double"); 
        TNode._Field = TNode.addType("%Field", "Field"); 
        TNode._INTTYPE = TNode.addType("INTTYPE", "int");
        TNode._integer = TNode.addType("integer", "int"); 
        TNode._float = TNode.addType("float", "float");
        TNode._Path = TNode.addType("%Path", "Path"); 
        // of terminating a function
        //_String = addType("String" "String"); fixme, does this type appears in original proof ?

        // Predefined variables name
        // variables used by the old proof system and that we still need
        TNode.addName("ecReturn", "%Path", "ecReturn");
        TNode.addName("ecThrow", "%Path", "ecThrow");
        TNode.addName("XRES", "%Reference", "XRes");
    }
    
    
    /*
     * (non-Javadoc)
     * @see escjava.vcGeneration.ProverType#addTypeInfo(escjava.translate.InitialState, javafe.ast.Expr)
     */
    public Expr addTypeInfo(InitialState initState, Expr tree) {
        //FIXME Our prover logic is (presumably?) typed, so why do we need to do this here?
        tree = GC.implies(initState.getInitialState(), tree);
        return tree;
    }
    
    /*
     * (non-Javadoc)
     * @see escjava.vcGeneration.ProverType#rewrite(escjava.vcGeneration.TNode)
     */
    public TNode rewrite(TNode tree) {
    	TRoot root;
    	if (!(tree instanceof TRoot)) {
    		TDisplay.warn(
    				"For coq pretty printer, the node should be a root node! instead of "
    				+ tree.getClass());
    		root = new TRoot();
    		root.sons.add(tree);
    		tree.parent = root;
    	}
    	else {
    		root = (TRoot) tree;
    	}
    	tree = simplifyProofObligation(root);
        return tree;
    }
    
    
    /**
     * Rename the type, ie. the name {@link TypeInfo#old} to
     * a proper Coq name in the variable {@link TypeInfo#def} 
     * @param t the type name to rename.
     */
    private void coqRename(TypeInfo t){

        if(t.old.equals("null") )
      	  t.def = "Reference";
        else {
        	// common rules here //fixme, be more specific maybe
        	if(t.old.startsWith("java.")) //check if in the form java.x.y 
        		t.def = t.old.replace('.','_');
        	else {
        		TDisplay.err("Type not handled in "+t.old + 
        				"Considering it as a user defined type... ie Types");
        		t.def = "ReferenceType";
        	}
        }
    }
    
    /**
     * Returns a valid Coq name corresponding to the given VariableInfo.
     * @param vi the variable info to analyse.
     */
    public /*@ non_null @*/ String getVariableInfo(VariableInfo vi){ 	
    	if(vi.type != TNode._Type){
    		if(vi.def == null)
    			coqRename(vi);
    		
    		return vi.def;
		}
    	else {
    		/* 
    		 * When variable is a type, we first have to check if it's in the type table.
    		 * If yes, we take the name here (it's the case with predefined types like INTYPE, integer, Reference etc...
    		 * Else it's normally a user defined Type
    		 */
			    
    		Set keySet = TNode.typesName.keySet();
    		Iterator iter = keySet.iterator();
    		TypeInfo tiTemp = null;
    		String keyTemp = null;
    		
    		while(iter.hasNext()) {
    			try {
				    keyTemp = (String)iter.next();
				} catch (Exception e){
				    System.err.println(e.getMessage());
				}
				tiTemp = (TypeInfo)TNode.typesName.get(keyTemp);
				if(tiTemp.old.equals(vi.old)) {
				    return getTypeInfo(tiTemp);
				}
				    
    		}
    		TDisplay.warn("considering "+ vi.old +
    				" as a user defined type, or a not (yet) " +
    				"handled variable.");
    			
    		coqRename(vi);
    		
    		return vi.def;
    	}
    }
	
    /*
     * (non-Javadoc)
     * @see escjava.vcGeneration.ProverType#labelRename(java.lang.String)
     */
	public String labelRename(String label) {
        label = label.replace('.','_');
        label = label.replace('<','_');
        label = label.replace('>','_');
        return label;
    }
    
    /** 
     * Rename the variable, ie. the name {@link VariableInfo#old} to
     * a proper Coq name in the variable {@link VariableInfo#def}.
     * Stolen from the PVS plugin...
     * @param vi the variable to rename.
     */
    private void coqRename(VariableInfo vi){

    	String coq;
		// definitions of different regexp used.
		Pattern p1 = Pattern.compile("\\|(.*):(.*)\\.(.*)\\|");
		Matcher m1 = p1.matcher(vi.old);
		
		// <variables not handled>
		Pattern pUnknownTag = Pattern.compile("Unknown tag <.*>");
		Matcher mUnknownTag = pUnknownTag.matcher(vi.old);
		
		Pattern pBrokenObject = Pattern.compile("\\|brokenObj<.*>\\|");
		Matcher mBrokenObject = pBrokenObject.matcher(vi.old);
		// </variables not handled>
		Pattern p4 = Pattern.compile("\\|(.*):(.*)\\|");
		Matcher m4 = p4.matcher(vi.old);
		String name = null;
		String line = null;
		String column = null;

		int i = 0;
		

		//System.out.println("Performing regular expression matching against "+old+" ");

		/*
		  This case is done first because some variables not handled are "%Type" ones
		  and thus otherwise they will be matched by the next if construct
		*/
		if(mUnknownTag.matches() || mBrokenObject.matches() || vi.old.equals("brokenObj")) { // variable is not handled yet, ask David Cok about
		    // some things
		    coq = "(* %NotHandled *) brokenObj"; 
		    vi.type = TNode._Reference;
		    
		}

		// <case 1>
		/*
		  If variable's a type, we must prefix his renaming by userDef?.
		  Why ?
		  Because if the user defined a type named 'Reference', otherwise we will use is as it
		  in the proof and it will interfer with our predefined types.

		  Since '?' isn't a valid character in the java notation and is Ok for pvs,
		  we use it to make the difference.
		*/
		else if(vi.type == TNode._Type){
		    TDisplay.warn("considering "+vi.old+" as a user defined type.");

		    // renaming done here
		    coq = "userDef_"+ vi.old;
		}
		// </case 1>

		// <case 2>, capturing |y:8.31|
		else if(m1.matches()){
		    //@ assert m1.groupCount() == 3;

		    if(m1.groupCount() != 3) {
		    	TDisplay.err("m.groupCount() != 3");
		    }
		    for( i = 1; i <= m1.groupCount(); i++) {

		    	if(m1.start(i) == -1 || m1.end(i) == -1) {
		    		TDisplay.err("return value of regex matching is -1");
		    	}
		    	else {
			    
		    		String temp = vi.old.substring(m1.start(i),m1.end(i));
		    		//@ assert temp != null;

		    		switch(i){ 
		    			case 1 :
		    				name = temp;
		    				//@ assert name != null;
		    				break;
		    			case 2:
		    				line = temp;
		    				//@ assert line != null;
		    				break;
		    			case 3:
		    				column = temp;
		    				//@ assert column != null;
		    				break;
		    			default :
		    				TDisplay.err("switch call incorrect, switch on value "+i);
		    		}
		    	} // no error in group			
		    } // checking all group

		    /* renaming done here, if you want the way it's done
		       (for pattern like |y:8.31|)
		       do it here. */
		    coq = name+"_"+line+"_"+column;
		} // </case 2>

		else if(m4.matches()) {
		    if(m4.groupCount() != 2)
				TDisplay.err("m.groupCount() != 3");

		    for( i = 1; i <= m4.groupCount(); i++) {

				if(m4.start(i) == -1 || m4.end(i) == -1)
				    TDisplay.err("return value of regex matching is -1");
				else {
				    
				    String temp = vi.old.substring(m4.start(i),m4.end(i));
				    //@ assert temp != null;

				    switch(i){ 
				    	case 1 :
				    		name = temp;
				    		//@ assert name != null;
				    		break;
				    	case 2:
				    		line = temp;
				    		//@ assert line != null;
				    		break;
				    	default :
				    		TDisplay.err("switch call incorrect, switch on value "+i);
				    }
				} // no error in group
			
		    } // checking all group

			    
		    if(line.equals("unknown"))
		    	coq = name;
		    else
		    	coq = name+"_"+line;
		}
		else {
		    coq = vi.old; // FIXME handle everything
		} // regexp didn't match
		coq = removeInvalidChar(coq);

		if(coq.equals("Z"))
			coq = "z";
		vi.def = coq;
    }
	
    /**
     * Removes the invalid characters in Coq for a given String.
     * @param name The name where some invalid characters appear.
     * @return The name without the invalid characters.
     */
    public String removeInvalidChar (String name) {
    	name = name.replace('@', '_');
		name = name.replace('#', '_');
		name = name.replace('.', '_');
		name = name.replace('-', '_');
		name = name.replace('<', '_');
		name = name.replace('>', '_');
		name = name.replace('?', '_');
		name = name.replace('[', '_');
		name = name.replace(']', '_');
		name = name.replace('!', '_');
		name = name.replaceAll("\\|", "");
		return name;
    }
    
    
    /**
     * Generates the declarations for the forall construct in the Coq Fashion,
     * i.e. (var1: Type1) (var2 : Type2)...
     * @param s The output
     * @param variablesNames the variables' names.
     */
    public void generateDeclarations(/*@ non_null @*/Writer s, HashMap variablesNames) {
    	Set keySet = variablesNames.keySet();

        Iterator iter = keySet.iterator();
        String keyTemp = null;
        VariableInfo viTemp = null;
        // il y a des limites a la grossierete. on vire le HashMap et on le remplace par un HashSet.
        while (iter.hasNext()) {

            try {
                keyTemp = (String) iter.next();
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            viTemp = (VariableInfo) variablesNames.get(keyTemp);

            String name = viTemp.getVariableInfo().toString();
            if (name.equals("ecReturn") || name.equals("ecThrow")
            		|| name.equals("allocNew_")
            		|| name.equals("alloc_")
                    || name.equals("int"))
                continue;
            
            // only writes variables with a known type ;)
            if (viTemp.type != null) {
            	try {
            		s.write("(");
            		s.write(viTemp.getVariableInfo() + " : "
            				+ viTemp.type.getTypeInfo());
            		s.write(")");
            	}
            	catch (IOException e) {
            		e.printStackTrace();
            	}

            } 
            else {
            	TDisplay.err("I won't write the variable " + viTemp.getVariableInfo() +" because I've got no idea of its type.");
            }
        }
    }
}
