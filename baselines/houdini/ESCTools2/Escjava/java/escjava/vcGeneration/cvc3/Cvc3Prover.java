package escjava.vcGeneration.cvc3;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.Vector;
import java.io.*;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import javafe.ast.Expr;
import javafe.util.ErrorSet;

import escjava.translate.GC;
import escjava.translate.InitialState;
import escjava.vcGeneration.*;

public class Cvc3Prover extends ProverType {

    // label to varname hash
    HashMap lablelhash = new HashMap();
    // a counter to ensure unique names
    int labelcounter = 0;

    int classCounter = 1;
    HashMap classType = new HashMap();
   
    // how to pass labels to the prover 
    // since cvc labels are actually abstract variables, we will
    // map the label to a (fresh) variable name cvc#labelvar
    public String labelRename(String label) {
// possibly use once we figure out labels
/*
        if (labelhash.containsKey(label)) {
          return labelhash(get(label)).toString();
        } else {
          String s = "cvc"+labelcounter+"labelvar";
          labelhash.put(label,s);
`          labelcounter++;
          return s;
        }
*/
        return "[label:"+label+"]";
    }
    
    // cvc3's visitor
    public TVisitor visitor(Writer out) {
        return new TCvc3Visitor(out,this);
    }
    
    // turn the ast into a cvc formula to be asserted/queried
    public void getProof(Writer out, String proofName, TNode term) {
    	try {     		
    		out.write("INCLUDE \"header.cvc3\";\n");
    		generateDeclarations(out,term);
    		generatePureMethodDeclarations(out);
    		out.write("QUERY\n");
    		generateTerm(out,term);
    		out.write(";");
    	} catch (IOException e) {
    		ErrorSet.fatal("internal error: " + e.getMessage());
    	}
    }
    

    // any variable names 
    public/*@ non_null @*/String getVariableInfo(VariableInfo caller) {

        if (caller.type != TNode._Type) {
            if (caller.def == null)
                cvc3Rename(caller);

            return caller.def;
        } else { // deal with (potentially new) type decls
            Set keySet = TNode.typesName.keySet();
            Iterator iter = keySet.iterator();
            TypeInfo tiTemp = null;
            String keyTemp = null;
            while (iter.hasNext()) {
              try {
                keyTemp = (String) iter.next();
              } catch (Exception e) {
                System.err.println(e.getMessage());
              }
              tiTemp = (TypeInfo) TNode.typesName.get(keyTemp);
                
              if (tiTemp.old.equals(caller.old)) {
                return getTypeInfo(tiTemp);
              }

            }
            TDisplay.warn("considering"+caller.old
              +"as a new user defined type or not yet handled variable.");
            cvc3Rename(caller);
            return caller.def;
        }
    }

    // renames any identifiers to be cvc-friendly
    // cvc identifiers are [A-Za-z][A-Za-z0-9'_\]*
    // so we need to rename any identifers that do not begin with a letter
    // or contain [%|.]
    // we map % -> '
    //        . -> '
    //        : -> '
    // and add 'x' to the beginning of all names that start with either a
    // non-letter or 'x' (so there is no conflict with names already 
    // starting with 'x')
    // NOTE! to ensure that we never recursively rename something, always
    // idRename against the .old value, not the .def
    private String cvc3idRename (String s) {
      s = s.replace('%','\'');
      s = s.replace('.','\'');
      s = s.replace(':', '\'');
      s = s.replace('?', '\'');
      s = s.replace('$', '\'');
      // replace other characters with underscores (could be done better)
      s = s.replace('|','_');
      s = s.replace('@', '_');
      s = s.replace('#', '_');
      s = s.replace('-', '_');
      s = s.replace('<', '_');
      s = s.replace('>', '_');
      s = s.replace('[', '_');
      s = s.replace(']', '_');
      s = s.replace('!', '_');

      Pattern p0 = Pattern.compile("^[x0-9\\\'_].*");
      Matcher m0 = p0.matcher(s);
      if (m0.matches()) {
        s = "x"+s;
      }
                                                                                      return s;
    }

    /*! ensures cvc != null; !*/
    // renames variables
    // for completeness, we should also check for keword conflicts (not done)
    private void cvc3Rename(VariableInfo caller) {

        // <variables not handled>
        Pattern p2 = Pattern.compile("Unknown tag <.*>");
        Matcher m2 = p2.matcher(caller.old);

        Pattern p3 = Pattern.compile("\\|brokenObj<.*>\\|");
        Matcher m3 = p3.matcher(caller.old);
        // </variables not handled>

        /*
         These should be variables local to some quantifier.
         Wheee...  It appears that they are declared as "Unknown tags" and
         referred to as "brokenObjs".
         We will rename them to "not_handled" and then clean up this mess later.
         */
        if (m2.matches() || m3.matches() || caller.old.equals("brokenObj")) { 
            caller.def = "not_handled";
        } 
        else
          caller.def = cvc3idRename(caller.old);
    }

    public String getClassTypeValue(String classname) {
      String s;
      if (classType.containsKey(classname))
        s = (String)classType.get(classname);
      else {
        s = "javaClass("+(classCounter++)+")";
        classType.put(classname,s);
      }
      return s;
    }

    public String getTypeInfo(TypeInfo caller) {
        if (caller.def == null)
            cvc3Rename(caller);

        return caller.def;
    }

    // for completeness, we should also check for keword conflicts (not done)
    // rename types
    /*! ensures cvc != null; !*/
    private void cvc3Rename(/*@ non_null @*/TypeInfo caller) {
        if (caller.old.equals("null")) {
          caller.def = "NULL!JavaValue";
        } else if (caller.old.equals("java.lang.Object")) {
          caller.def = "Object";
        } else { 
// do we only want to allow java.x.y?
          // follow rules for variable renaming, above
          caller.def = cvc3idRename(caller.old);
        }
    }

    public void init() {
        // Predefined types

        TNode._Reference = TNode.addType("%Reference", "ReferenceValue");
        TNode._Time = TNode.addType("%Time", "Time");
        TNode._Type = TNode.addType("%Type", "JavaType");
        TNode._boolean = TNode.addType("boolean", "javaBool");
        TNode._char = TNode.addType("char", "javaChar");
        TNode._DOUBLETYPE = TNode.addType("DOUBLETYPE", "javaDouble");
        TNode._double = TNode.addType("double", "javaDouble");
        TNode._Field = TNode.addType("%Field", "Field"); // fixme there's a lot of different fields in the pvs logic, I need to capture that
        TNode._INTTYPE = TNode.addType("INTTYPE", "javaInt"); //fixme like DOUBLETYPE
        TNode._integer = TNode.addType("integer","javaInt"); //fixme
        TNode._float = TNode.addType("float", "javaFloat");
        TNode._Path = TNode.addType("%Path", "Path"); // used to modelize different ways
        // of terminating a function
        //_String = addType("String", "?"); fixme, does this type appears in original proof ?

        // Predefined variables name
        // variables used by the old proof system and that we still need
        TNode.addName("ecReturn", "%Path", "ecReturn");
        TNode.addName("ecThrow", "%Path", "ecThrow");
        TNode.addName("XRES", "%Reference", "XRES");        
        TNode.addName("java.lang.Object", "JavaType", "Object");        
        TNode.addName("java'lang'Object", "JavaType", "Object");        
        TNode.addName("java.lang.Exception", "JavaType", "Exception");        
        TNode.addName("alloc", "Time", "alloc");        
        TNode.addName("alloc_", "Time", "alloc_");        
        TNode.addName("LS", "LockSet", "LS");        
    }
    
    public Expr addTypeInfo(InitialState initState, Expr tree) {
        tree = GC.implies(initState.getInitialState(), tree);
        return tree;
    }

    public TNode rewrite(TNode tree) {
// worry about this later
        // Intentionally do nothing!
        return tree;
    }

    // Generate declarations for pure methods (later used with "MethodCall" nodes
    // adapted from coq code
    public void generatePureMethodDeclarations (Writer s) {
    	Vector methNames = TMethodCall.methNames;
    	Vector methDefs = TMethodCall.methDefs;
    	try {
    		for (int i = 0; i<methNames.size();i++) {
    			s.write(getVariableInfo(((VariableInfo)methNames.get(i)))+ " : ");
    			TMethodCall tmc = (TMethodCall)(methDefs.get(i));
    			Vector v = tmc.getArgType();
    			int j=0;
    			if (v.size() > 1)
    				s.write("(");
    			for (;j<v.size();j++) {
    				TypeInfo ti = (TypeInfo)v.get(j);
    				if (ti == null) 
    					s.write("JavaValue");
    				else
    					s.write(getTypeInfo(ti));
    				
    				if (j < v.size()-1)
    					s.write (",");
    			}
    			if (v.size() > 1)
    				s.write(")");
    			if (tmc.type == null) 
    				s.write (" -> JavaValue;\n");
    			else
    				s.write (" -> " + tmc.type.def + ";\n");
    		}
    	} catch (IOException e) {
    		ErrorSet.fatal("internal error: " + e.getMessage());
    	}
    }
    
    // so this goes through the map of variable info and prints out type decls.
//  this will probably not work with local (quantified) variables...
//  may need to rename those
    public void generateDeclarations(/*@ non_null @*/ Writer s, HashMap variablesNames) {
    	try {
    		Set keySet = variablesNames.keySet();
    		Iterator iter = keySet.iterator();
    		String keytmp = null;
    		VariableInfo vitmp = null;
    		HashMap varhash = new HashMap();
    		// include pre-defined vars
//  		varhash.put("ecReturn",null);
//  		varhash.put("ecThrow",null);
    		varhash.put("alloc",null);
    		varhash.put("alloc_",null);
    		varhash.put("LS",null);
    		varhash.put("elems",null);
    		varhash.put("javaInt",null);
    		varhash.put("javaChar",null);
    		varhash.put("javaDouble",null);
    		varhash.put("javaBool",null);
    		varhash.put("javaFloat",null);
    		varhash.put("java'lang'Object",null); // base Object type
    		varhash.put("Object",null); // base Object type
    		varhash.put("Exception",null); // Exception class type
    		varhash.put("not_handled",null); // this will be handled in the visitor
    		
    		while (iter.hasNext()) {
    			try {
    				keytmp = (String) iter.next();
    			} catch (Exception e) {
    				System.err.println(e.getMessage());
    			}
    			
    			vitmp = (VariableInfo) variablesNames.get(keytmp);
//  			from coq -- ??
    			String name = vitmp.getVariableInfo().toString();
    			
//  			make sure we don't re-define anything!
    			if (varhash.containsKey(name))
    				continue;
    			varhash.put(name,vitmp);
    			
    			if (vitmp.type != null) {
    				// declare variable
    				s.write(vitmp.getVariableInfo() + " : "
    						+ vitmp.type.getTypeInfo());
    				// declare new object types
    				if (vitmp.type.getTypeInfo().equals("JavaType"))
    					s.write(" = "+getClassTypeValue(vitmp.getVariableInfo()));
    				
    				s.write(";\n");
//  				only assert the types of javatype vars!!
    				// now add static type decl:
//  				s.write("  ASSERT s_type("+vitmp.getVariableInfo()+")="
//  				+vitmp.type.getTypeInfo());
//  				s.write(";\n");
    				// dynamic type decl?
//  				s.write("  ASSERT d_type("+vitmp.getVariableInfo()+")="
//  				+vitmp.type.getTypeInfo());
//  				s.write(";\n");
    				
    			} else {
    				s.write(vitmp.getVariableInfo() + " : S;\n");
    				// static type decl
//  				s.write("  ASSERT s_type("+vitmp.getVariableInfo()+")= S");
//  				s.write(";\n");
    				// dynamic type decl?
    				
    				TDisplay.warn("Type of variable " + keytmp
    						+ " is not set when declarating variables for the proof, skipping it...");
    			}
    		}
    	} catch (IOException e) {
    		ErrorSet.fatal("internal error: " + e.getMessage());
    	}
    }
}
