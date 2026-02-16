/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.ast;

import java.io.OutputStream;
import java.util.Enumeration;

import javafe.ast.BinaryExpr;
import javafe.ast.DelegatingPrettyPrint;
import javafe.ast.Expr;
import javafe.ast.ExprVec;
import javafe.ast.FieldDecl;
import javafe.ast.GenericVarDecl;
import javafe.ast.GenericVarDeclVec;
import javafe.ast.Identifier;
import javafe.ast.LexicalPragma;
import javafe.ast.LiteralExpr;
import javafe.ast.LocalVarDecl;
import javafe.ast.ModifierPragma;
import javafe.ast.ModifierPragmaVec;
import javafe.ast.PrettyPrint;
import javafe.ast.StmtPragma;
import javafe.ast.Type;
import javafe.ast.TypeDeclElemPragma;
import javafe.ast.VarInit;
import javafe.ast.VariableAccess;
import javafe.util.Assert;
import escjava.ParsedRoutineSpecs;
import escjava.parser.EscPragmaParser;

public class EscPrettyPrint extends DelegatingPrettyPrint
{
  //@ ensures this.self == this;
  //@ ensures_redundantly this.del != null;
  public EscPrettyPrint() { }
  
  //@ requires del != this;
  //@ ensures this.self == self;
  //@ ensures this.del == del;
  public EscPrettyPrint(/*@non_null*/ PrettyPrint self, /*@non_null*/ PrettyPrint del) {
    super(self, del);
  }
  
  public void print(/*@non_null*/ OutputStream o, /*@ non_null */ LexicalPragma lp) {
    if (lp.getTag() == TagConstants.NOWARNPRAGMA) {
      write(o, "//@ ");
      write(o, TagConstants.toString(TagConstants.NOWARN));
      NowarnPragma nwp = (NowarnPragma)lp;
      for (int i = 0; i < nwp.checks.size(); i++) {
        if (i == 0) write(o, ' ');
        else write(o, ", ");
        write(o, nwp.checks.elementAt(i).toString());
      }
      write(o, "\n");
    } else writeln(o, "// Unknown LexicalPragma (tag = " + lp.getTag() + 
                   " " + TagConstants.toString(lp.getTag()) + ')');
  }
  
  public void exsuresPrintDecl(/*@non_null*/OutputStream o, /*@ nullable */ GenericVarDecl d) {
    if (d == null)
      write(o, "<null GenericVarDecl>");
    else {
      self.print(o, d.type);
      if (!(d.id.equals(TagConstants.ExsuresIdnName))) {
        write (o, ' ');
        write(o, d.id.toString());
      }
    }
  }
  
  public void print(/*@non_null*/ OutputStream o, 
		    int ind, 
		    /*@ non_null */ TypeDeclElemPragma tp) 
  {
    int tag = tp.getTag();
    int otag = tag; if (tp.isRedundant()) otag = TagConstants.makeRedundant(tag);
    switch (tag) {
    case TagConstants.AXIOM:
    case TagConstants.INVARIANT:
    case TagConstants.CONSTRAINT: {
      Expr e = ((ExprDeclPragma)tp).expr;
      write(o, "/*@ "); 
      write(o, TagConstants.toString(otag));
      write(o, ' ');
      self.print(o, ind, e);
      write(o, "; */");
      break;
    }
      
    case TagConstants.REPRESENTS: {
      Expr e = ((NamedExprDeclPragma)tp).expr;
      write(o, "/*@ "); 
      write(o, TagConstants.toString(otag));
      write(o, ' ');
      if(e instanceof BinaryExpr) {
	  BinaryExpr be = (BinaryExpr)e;
	  self.print(o, ind, be.left);
	  write(o, ' ');
	  write(o, TagConstants.toString(TagConstants.LEFTARROW));
	  write(o, ' ');
	  self.print(o, ind, be.right);
      } else {
	  // System.err.println("\tEscPrettyPrint: unexpected type: " + e);
	  self.print(o, ind, e);
      }
      write(o, "; */");
      break;
    }

    case TagConstants.MODELTYPEPRAGMA: {
      ModelTypePragma mtp = (ModelTypePragma)tp;
      write(o, "/*@ model ");
      self.print(o, ind, mtp.decl);
      write(o, "@*/");
        
      break;
    }
      
    case TagConstants.MODELDECLPRAGMA: {
      FieldDecl d = ((ModelDeclPragma)tp).decl;
      /*
       * Below is a "//@" to prevent illegal nested /* ...  comments
       * that otherwise might result from any attached modifier pragmas.
       *
       * We rely on the fact that no ESC modifier can generate newlines
       * when pretty printed.  !!!!
       */
      write(o, "//@ model ");
      self.print(o, ind, d, true); 
      // write(o, "  */\n");
      write(o, "\n");
      break;
    }
      
    case TagConstants.GHOSTDECLPRAGMA: {
      FieldDecl d = ((GhostDeclPragma)tp).decl;
      /*
       * Below is a "//@" to prevent illegal nested /* ...  comments
       * that otherwise might result from any attached modifier pragmas.
       *
       * We rely on the fact that no ESC modifier can generate newlines
       * when pretty printed.  !!!!
       */
      write(o, "//@ ghost ");
      self.print(o, ind, d, true); 
      // write(o, "  */\n");
      write(o, "\n");
      break;
    }
    case TagConstants.STILLDEFERREDDECLPRAGMA: {
      Identifier v = ((StillDeferredDeclPragma)tp).var;
      write(o, "/*@ ");
      write(o, TagConstants.toString(TagConstants.STILL_DEFERRED));
      write(o, " "); 
      write(o,v.toString()); 
      write(o, "; */");
      break;
    }
    default:
      write(o, "/* Unknown TypeDeclElemPragma (tag = " + TagConstants.toString(tag) + ") */");
      break;
    }
  }
  
  public void print(/*@non_null*/OutputStream o, int ind, /*@non_null*/ModifierPragmaVec v) {
    int n = v.size();
    for (int i=0; i<n; ++i) {
      print(o,ind,v.elementAt(i));
    }
  }
  
  public void print(/*@non_null*/OutputStream o, int ind, /*@ non_null */ ModifierPragma mp) {
    int tag = mp.getTag();
    switch (tag) {
    case TagConstants.ALSO:
      write(o, "/*@ "); 
      write(o, TagConstants.toString(mp.originalTag())); 
      write(o, " */");
      break;
      
    case TagConstants.OPENPRAGMA:
      writeln(o, "{|");
      ++ind;
      break;
      
    case TagConstants.CLOSEPRAGMA:
      --ind;
      writeln(o, "|}");
      break;
      
    case TagConstants.MODEL_PROGRAM:
      write(o, "/*@ "); 
      write(o, TagConstants.toString(tag)); 
      write(o, "{...} */"); // FIXME - do all of model program
      break;
      
    case TagConstants.ALSO_REFINE:
    case TagConstants.CODE_BIGINT_MATH:
    case TagConstants.CODE_CONTRACT:
    case TagConstants.CODE_JAVA_MATH:
    case TagConstants.CODE_SAFE_MATH:
    case TagConstants.END:
    case TagConstants.EXAMPLE:
    case TagConstants.EXCEPTIONAL_EXAMPLE:
    case TagConstants.FOR_EXAMPLE:
    case TagConstants.HELPER:
    case TagConstants.IMMUTABLE:
    case TagConstants.IMPLIES_THAT:
    case TagConstants.INSTANCE:
    case TagConstants.NULLABLE:
    case TagConstants.MONITORED:
    case TagConstants.NON_NULL:
    case TagConstants.NON_NULL_BY_DEFAULT: // Chalin-Kiniry experiments
    case TagConstants.NORMAL_EXAMPLE:
    case TagConstants.NULLABLE_BY_DEFAULT: // Chalin-Kiniry experiments
    case TagConstants.OBS_PURE: // Chalin-Kiniry experiments
    case TagConstants.PEER: // Universe type annotation
    case TagConstants.PURE:
    case TagConstants.SAME:
    case TagConstants.READONLY: // Universe type annotation
    case TagConstants.REP: // Universe type annotation
    case TagConstants.SPEC_BIGINT_MATH:
    case TagConstants.SPEC_JAVA_MATH:
    case TagConstants.SPEC_PROTECTED: // SC HPT AAST 3
    case TagConstants.SPEC_PUBLIC:
    case TagConstants.SPEC_SAFE_MATH:
    case TagConstants.UNINITIALIZED:
    case TagConstants.WRITABLE_DEFERRED:
      write(o, "/*@ "); 
      write(o, TagConstants.toString(tag)); 
      write(o, " */");
      break;
      
    case TagConstants.GHOST:
    case TagConstants.MODEL:
      break;
      
    case TagConstants.NO_WACK_FORALL:
    case TagConstants.OLD:
      write(o, "/*@ "); 
      write(o, TagConstants.toString(tag)); 
      LocalVarDecl d = ((VarDeclModifierPragma)mp).decl;
      self.print(o, ind, d, true); 
      write(o, "; */");
      break;
      
    case TagConstants.ALSO_ENSURES:
    case TagConstants.ALSO_REQUIRES:
    case TagConstants.ENSURES:
    case TagConstants.DIVERGES:
    case TagConstants.POSTCONDITION:
    case TagConstants.PRECONDITION:
    case TagConstants.WHEN:
    case TagConstants.MONITORED_BY: // from EscPragmaParser.continuePragma(Token)
    case TagConstants.READABLE_IF:
    case TagConstants.REQUIRES:
    case TagConstants.WRITABLE_IF: {
      write(o, "/*@ ");
      if (mp.isRedundant())
        write(o, TagConstants.toString(TagConstants.makeRedundant(tag)));
      else
        write(o, TagConstants.toString(tag)); 
      write(o, ' ');
      if (!(mp instanceof ExprModifierPragma)) System.out.print("{{{{ " + mp + "}}}}");
      Expr e = ((ExprModifierPragma)mp).expr;
      self.print(o, ind, e); 
      write(o, "; */");
      break;
    }
      
      // All redundant tokens should not exist in the AST
      // anymore; they are represented with redundant fields in
      // the AST nodes.
    case TagConstants.ASSERT_REDUNDANTLY:
    case TagConstants.ASSIGNABLE_REDUNDANTLY:
    case TagConstants.ASSUME_REDUNDANTLY:
    case TagConstants.CONSTRAINT_REDUNDANTLY:
    case TagConstants.DECREASES_REDUNDANTLY:
    case TagConstants.DECREASING_REDUNDANTLY:
    case TagConstants.DIVERGES_REDUNDANTLY:
    case TagConstants.DURATION_REDUNDANTLY:
    case TagConstants.ENSURES_REDUNDANTLY:
    case TagConstants.EXSURES_REDUNDANTLY:
    case TagConstants.HENCE_BY_REDUNDANTLY: 
    case TagConstants.INVARIANT_REDUNDANTLY: 
    case TagConstants.LOOP_INVARIANT_REDUNDANTLY: 
    case TagConstants.MAINTAINING_REDUNDANTLY:
    case TagConstants.MEASURED_BY_REDUNDANTLY:
    case TagConstants.MODIFIABLE_REDUNDANTLY:
    case TagConstants.MODIFIES_REDUNDANTLY:
    case TagConstants.POSTCONDITION_REDUNDANTLY:
    case TagConstants.PRECONDITION_REDUNDANTLY:
    case TagConstants.REPRESENTS_REDUNDANTLY:
    case TagConstants.REQUIRES_REDUNDANTLY:
    case TagConstants.SIGNALS_REDUNDANTLY:
    case TagConstants.WHEN_REDUNDANTLY:
    case TagConstants.WORKING_SPACE_REDUNDANTLY:
      Assert.fail("Redundant keywords should not be in AST!: "
                  + TagConstants.toString(tag));
      break;
      
    case TagConstants.MODIFIESGROUPPRAGMA: {
      ModifiesGroupPragma mm = (ModifiesGroupPragma)mp;
      write(o, "/*@ modifies ");
      if (mm.precondition != null) {
        self.print(o,ind,mm.precondition);
        write(o, " ==> (");
      }
      for (int i=0; i<mm.items.size(); ++i) {
        if (i != 0) write(o, ", ");
        CondExprModifierPragma ce = mm.items.elementAt(i);
        self.print(o, ind, ce.expr); 
        if (ce.cond != null) {
          write(o, " if ");
          self.print(o, ind, ce.cond); 
        }
      }
      if (mm.precondition != null) write(o, ")");
        
      write(o, "; @*/");
      break;
    }
      
    case TagConstants.DURATION:
    case TagConstants.WORKING_SPACE:
    case TagConstants.ALSO_MODIFIES:
    case TagConstants.ASSIGNABLE:
    case TagConstants.ONLY_ASSIGNED:
    case TagConstants.MEASURED_BY:
    case TagConstants.MODIFIABLE:
    case TagConstants.MODIFIES: {
      Expr e = ((CondExprModifierPragma)mp).expr;
      Expr p = ((CondExprModifierPragma)mp).cond;
      write(o, "/*@ "); 
      if (mp.isRedundant())
        write(o, TagConstants.toString(TagConstants.makeRedundant(tag))); 
      else
        write(o, TagConstants.toString(tag)); 
      write(o, ' ');
      self.print(o, ind, e); 
      if (p != null) {
        write(o, " if ");
        self.print(o, ind, p); 
      }
      write(o, "; */");
      break;
    }
      
    case TagConstants.ALSO_EXSURES: 
    case TagConstants.EXSURES:
    case TagConstants.SIGNALS: {
      VarExprModifierPragma vemp = (VarExprModifierPragma)mp;
      write(o, "/*@ "); 
      write(o, TagConstants.toString(mp.originalTag()));
      write(o, " ("); 
      //self.print(o, vemp.arg);
      exsuresPrintDecl(o, vemp.arg); 
      write(o, ") ");
      self.print(o, ind, vemp.expr); 
      write(o, "; */");
      break;
    }
      
    case TagConstants.BEHAVIOR:
    case TagConstants.EXCEPTIONAL_BEHAVIOR:
    case TagConstants.NORMAL_BEHAVIOR:
      write(o, "/*@ "); 
      write(o, TagConstants.toString(tag));
      write(o, " */");
      break;
      
    case TagConstants.PARSEDSPECS: {
      ParsedRoutineSpecs s = ((ParsedSpecs)mp).specs;
      if (s.initialAlso != null) self.print(o,ind,s.initialAlso);
      java.util.Iterator i = s.specs.iterator();
      int k = 0;
      while (i.hasNext()) {
        Object oo = i.next();
        if (oo == mp || oo == s) break;
        print(o,ind,(ModifierPragmaVec)oo);
      }
      if (s.impliesThat.size() > 0) writeln(o, "/*@ implies_that */"); 
      i = s.impliesThat.iterator();
      while (i.hasNext()) {
        print(o,ind,(ModifierPragmaVec)i.next());
      }
      if (s.examples.size() > 0) writeln(o, "/*@ for_example */");
      i = s.examples.iterator();
      while (i.hasNext()) {
        print(o,ind,(ModifierPragmaVec)i.next());
      }
      for (int j=0; j<s.modifiers.size(); ++j) {
        //print(o,ind,s.modifiers.elementAt(j));
      }
      break;
    }
      
    default:
      write(o, "/* Unknown ModifierPragma (tag = " + TagConstants.toString(tag) + ") */");
      break;
    }
  }
  
  public void print(/*@non_null*/OutputStream o, int ind, /*@ non_null */ StmtPragma sp) {
    int tag = sp.getTag();
    int otag = sp.originalTag();
    switch (tag) {
    case TagConstants.UNREACHABLE:
      write(o, "/*@ "); 
      write(o, TagConstants.toString(otag)); 
      write(o, " */");
      break;
      
    case TagConstants.ASSERT: {
      Expr e = ((ExprStmtPragma)sp).expr;
      Expr l = ((ExprStmtPragma)sp).label;
      write(o, "/*@ "); 
      write(o, TagConstants.toString(otag)); 
      write(o, " ");
      self.print(o, ind, e);
      if (l != null) {
        write(o, " : ");
        self.print(o, ind, l);
      }
      write(o, "; */");
      break;
    }
      
    case TagConstants.HENCE_BY:
    case TagConstants.ASSUME:
    case TagConstants.DECREASES:
    case TagConstants.DECREASING:
    case TagConstants.MAINTAINING:
    case TagConstants.LOOP_INVARIANT:
    case TagConstants.LOOP_PREDICATE: {
      Expr e = ((ExprStmtPragma)sp).expr;
      write(o, "/*@ "); 
      write(o, TagConstants.toString(otag)); 
      write(o, ' ');
      self.print(o, ind, e); 
      write(o, "; */");
      break;
    }
      
    case TagConstants.SETSTMTPRAGMA: {
      SetStmtPragma s = (SetStmtPragma)sp;
      write(o, "/*@ ");
      write(o, TagConstants.toString(TagConstants.SET));
      write(o, " ");
      self.print(o, ind, s.target);
      write(o, " = ");
      self.print(o, ind, s.value);
      write(o, "; */");
      break;
    }
      
    default:
      write(o, "/* Unknown StmtPragma (tag = " + TagConstants.toString(otag) + ") */");
      break;
    }
  }

  public static void print(/*@ nullable */ GuardedCmd g) {
    ((EscPrettyPrint)inst).print(System.out,0,g);
  }
  
  /** Print a guarded command.  Assumes that <code>g</code> should be
      printed starting at the current position of <code>o</code>.  It
      does <em>not</em> print a new-line at the end of the statement.
      However, if the statement needs to span multiple lines (for
      example, because it has embedded statements), then these lines are
      indented by <code>ind</code> spaces. */
  
  public void print(/*@non_null*/ OutputStream o, int ind, /*@ nullable */ GuardedCmd g) {
    if (g == null) {
      writeln(o, "<null Stmt>");
      return;
    }
    
    int tag = g.getTag();
    switch (tag) {
    case TagConstants.SKIPCMD:
      write(o, "SKIP");
      return;
      
    case TagConstants.RAISECMD:
      write(o, "RAISE");
      return;
      
    case TagConstants.ASSERTCMD:
      write(o, "ASSERT ");
      print(o, ind, ((ExprCmd)g).pred);
      return;
      
    case TagConstants.ASSUMECMD:
      write(o, "ASSUME ");
      print(o, ind, ((ExprCmd)g).pred);
      return;
      
    case TagConstants.GETSCMD: {
      GetsCmd gc = (GetsCmd)g;
      if (escjava.Main.options().nvu)
        write(o, gc.v.decl.id.toString());
      else
        write(o, escjava.translate.UniqName.variable(gc.v.decl));
      write(o, " = ");
      if (gc.rhs != null) print(o, ind, gc.rhs);
      return;
    }
      
    case TagConstants.SUBGETSCMD: {
      SubGetsCmd sgc = (SubGetsCmd)g;
      if (escjava.Main.options().nvu)
        write(o, sgc.v.decl.id.toString());
      else
        write(o, escjava.translate.UniqName.variable(sgc.v.decl));
      write(o, "[");
      print(o, ind, sgc.index);
      write(o, "] = ");
      if (sgc.rhs != null) print(o, ind, sgc.rhs);
      return;
    }
      
    case TagConstants.SUBSUBGETSCMD: {
      SubSubGetsCmd ssgc = (SubSubGetsCmd)g;
      if (escjava.Main.options().nvu)
        write(o, ssgc.v.decl.id.toString());
      else
        write(o, escjava.translate.UniqName.variable(ssgc.v.decl));
      write(o, "[");
      print(o, ind, ssgc.index1);
      write(o, "][");
      print(o, ind, ssgc.index2);
      write(o, "] = ");
      if (ssgc.rhs != null) print(o, ind, ssgc.rhs);
      return;
    }
      
    case TagConstants.RESTOREFROMCMD: {
      RestoreFromCmd gc = (RestoreFromCmd)g;
      write(o, "RESTORE ");
      if (escjava.Main.options().nvu)
        write(o, gc.v.decl.id.toString());
      else
        write(o, escjava.translate.UniqName.variable(gc.v.decl));
      write(o, " FROM ");
      print(o, ind, gc.rhs);
      return;
    }
      
    case TagConstants.VARINCMD: {
      VarInCmd vc = (VarInCmd)g;
      write(o, "VAR");
      printVarVec(o, vc.v);
      writeln(o, " IN");
      spaces(o, ind+INDENT);
      print(o, ind+INDENT, vc.g);
      writeln(o);
      spaces(o, ind);
      write(o, "END");
      return;
    }
      
    case TagConstants.DYNINSTCMD: {
      DynInstCmd dc = (DynInstCmd)g;
      writeln(o, "DynamicInstanceCmd \"" + dc.s + "\" IN");
      spaces(o, ind+INDENT);
      print(o, ind+INDENT, dc.g);
      writeln(o);
      spaces(o, ind);
      write(o, "END");
      return;
    }
      
    case TagConstants.SEQCMD: {
      SeqCmd sc = (SeqCmd)g;
      int len = sc.cmds.size();
      if (len == 0) write(o, "SKIP");
      else if (len == 1) print(o, ind, sc.cmds.elementAt(0));
      else {
        for (int i = 0; i < len; i++) {
          if (i != 0) {
            writeln(o, ";");
            spaces(o, ind);
          }
          print(o, ind, sc.cmds.elementAt(i));
        }
      }
      return;
    }
      
    case TagConstants.CALL: {
      Call call = (Call)g;
      if (call.inlined) {
        write(o, "INLINED ");
      }
      write(o, "CALL "+ call.spec.dmd.getId());
      print(o, ind, call.args );
      if (escjava.Main.options().showCallDetails) {
        writeln(o, " {");
        spaces(o, ind+INDENT);
        printSpec(o, ind+INDENT, call.spec );
        writeln(o);
        spaces(o, ind);
        writeln(o, "DESUGARED:");
        spaces(o, ind+INDENT);
        print(o, ind+INDENT, call.desugared);
        writeln(o);
        spaces(o, ind);
        write(o, "}");
      }
      return;
    }
      
    case TagConstants.TRYCMD: {
      CmdCmdCmd tc = (CmdCmdCmd)g;
      write(o, "{");
      spaces(o, INDENT-1 );
      print(o, ind+INDENT, tc.g1);
        
      writeln(o);
      spaces(o, ind);
      write(o, "!");
      spaces(o, INDENT-1 );
      print(o, ind+INDENT, tc.g2);
      writeln(o);
        
      spaces(o, ind);
      write(o, "}");
      return;
    }
      
    case TagConstants.LOOPCMD: {
      LoopCmd lp = (LoopCmd)g;
      writeln(o, "LOOP");
      printCondVec(o, ind+INDENT, lp.invariants,
                   TagConstants.toString(TagConstants.LOOP_INVARIANT));
      printDecrInfoVec(o, ind+INDENT, lp.decreases,
                       TagConstants.toString(TagConstants.DECREASES));
        
      int nextInd = ind+INDENT;
      if (lp.tempVars.size() != 0) {
        spaces(o, nextInd);
        write(o, "VAR");
        printVarVec(o, lp.tempVars);
        writeln(o, " IN");
        nextInd += INDENT;
      }
        
      spaces(o, nextInd);
      print(o, nextInd, lp.guard);
      // let a double-semicolon denote the break between the "guard" and "body"
      writeln(o, ";;");
      spaces(o, nextInd);
      print(o, nextInd, lp.body);
      writeln(o);
        
      if (lp.tempVars.size() != 0) {
        spaces(o, ind+INDENT);
        writeln(o, "END");
      }
        
      if (escjava.Main.options().showLoopDetails) {
        spaces(o, ind);
        writeln(o, "PREDICATES:");
        for (int i = 0; i < lp.predicates.size(); i++) {
          spaces(o, ind+INDENT);
          print(o, ind+INDENT, lp.predicates.elementAt(i));
          writeln(o);
        }
          
        /*
          spaces(o, ind);
          writeln(o, "INVARIANTS:");
          for (int i = 1; i < lp.invariants.size(); i++) {
          spaces(o, ind+INDENT);
          print(o, ind+INDENT, lp.invariants.elementAt(i).pred);
          writeln(o);
          }
        */
          
        spaces(o, ind);
        writeln(o, "DESUGARED:");
        spaces(o, ind+INDENT);
        print(o, ind+INDENT, lp.desugared);
        writeln(o);
      }
        
      spaces(o, ind);
      write(o, "END");
      return;
    }
      
    case TagConstants.CHOOSECMD: {
      CmdCmdCmd cc = (CmdCmdCmd)g;
      write(o, "{");
      spaces(o, INDENT-1 );
      print(o, ind+INDENT, cc.g1);
        
      writeln(o);
      spaces(o, ind);
      writeln(o, "[]");
        
      spaces(o, ind+INDENT);
      print(o, ind+INDENT, cc.g2);
      writeln(o);
        
      spaces(o, ind);
      write(o, "}");
      return;
    }
      
    default:
      write(o, "UnknownTag<" + tag + ":");
      write(o, TagConstants.toString(tag) + ">");
      return;
    }
  }
  
  private void printVarVec(/*@non_null*/ OutputStream o, /*@non_null*/ GenericVarDeclVec vars) {
    for (int i = 0; i < vars.size(); i++) {
      GenericVarDecl vd = vars.elementAt(i);
      write(o, ' ');
      if (false) {
        // Someday we may have special variables for map types
        write(o, "Map[");
        // write(o, ((FieldDecl)vd).parent.id.toString());
        write(o, " -> ");
        print(o, vd.type);
        write(o, "]");
      } else print(o, vd.type);
      write(o, ' ');
      if (escjava.Main.options().nvu)
        write(o, vd.id.toString());
      else
        write(o, escjava.translate.UniqName.variable(vd));
      if (i != vars.size()-1) {
        write(o, ';');
      }
    }
  }
  
  public void printDMD(/*@non_null*/OutputStream o, int ind, /*@non_null*/DerivedMethodDecl dmd) {
    write(o, "DMD ");
    
    ModifierPragmaVec local = dmd.original.pmodifiers;
    ModifierPragmaVec combined = ModifierPragmaVec.make();
    
    for (int i=0; i<dmd.requires.size(); i++)
      combined.addElement(dmd.requires.elementAt(i));
    for (int i=0; i<dmd.modifies.size(); i++)
      combined.addElement(dmd.modifies.elementAt(i));
    for (int i=0; i<dmd.ensures.size(); i++)
      combined.addElement(dmd.ensures.elementAt(i));
    for (int i=0; i<dmd.exsures.size(); i++)
      combined.addElement(dmd.exsures.elementAt(i));
    
    dmd.original.pmodifiers = combined;
    print(o, ind+INDENT, dmd.original,
          dmd.getContainingClass().id,
          false);
    dmd.original.pmodifiers = local;
    return;
  }
 
  public void printSpec(/*@non_null*/OutputStream o, int ind, /*@non_null*/Spec spec) {
    write(o, "SPEC ");
    printDMD(o,ind,spec.dmd);
    
    spaces(o, ind);
    write(o, "targets ");
    print(o, ind, spec.targets);
    writeln(o, ";");
    
    spaces(o, ind);
    write(o, "prevarmap { ");
    boolean first=true;
    for(Enumeration e = spec.preVarMap.keys(); e.hasMoreElements(); )
      {
        if( !first )
          write(o, ", ");
        else
          first = false;
      
        GenericVarDecl vd = (GenericVarDecl)e.nextElement();
        VariableAccess va = (VariableAccess)spec.preVarMap.get(vd);
        print( o, vd );
        write(o, "->" );
        print( o, ind, va );
      }
    writeln(o, " };");
    
    printCondVec(o, ind, spec.pre, "pre");
    printCondVec(o, ind, spec.post, "post");
    return;
  }
  
  public void printCondVec(/*@non_null*/OutputStream o, int ind, /*@non_null*/ConditionVec cv,
                           /*@non_null*/String name) {
    for(int i=0; i<cv.size(); i++)
      {
        spaces(o, ind);
        write(o, name+" ");
        printCond(o, ind, cv.elementAt(i));
        writeln(o, ";");
      }
  }
  
  public void printDecrInfoVec(/*@non_null*/OutputStream o, int ind,
                               /*@non_null*/DecreasesInfoVec div, /*@non_null*/String name) {
    for (int i = 0; i < div.size(); i++) {
      spaces(o, ind);
      write(o, name+" ");
      print(o, ind, div.elementAt(i).f);
      writeln(o, ";");
    }
  }
  
  public void printCond(/*@non_null*/OutputStream o, int ind, /*@non_null*/Condition cond) {
    write(o, TagConstants.toString(cond.label)+": ");
    print(o, ind, cond.pred );
  }
  
  public void print(/*@non_null*/OutputStream o, int ind, VarInit e) {
    int tag = e.getTag();
    switch (tag) {
      
    case TagConstants.VARIABLEACCESS: {
      VariableAccess lva = (VariableAccess)e;
      if (escjava.Main.options().nvu)
        write(o, lva.decl.id.toString());
      else
        write(o, escjava.translate.UniqName.variable(lva.decl));
      break;
    }
      
    case TagConstants.IMPLIES:
    case TagConstants.EXPLIES:
    case TagConstants.IFF:
    case TagConstants.NIFF:
    case TagConstants.SUBTYPE: {
      BinaryExpr be = (BinaryExpr)e;
      self.print(o, ind, be.left);
      write(o, ' '); write(o, TagConstants.toString(be.op)); write(o, ' ');
      self.print(o, ind, be.right);
      break;
    }
      
    case TagConstants.SYMBOLLIT:
      write(o, (String) (((LiteralExpr)e).value));
      break;
      
    case TagConstants.LABELEXPR: {
      LabelExpr le = (LabelExpr)e;
      write(o, "(");
      write(o, (le.positive ? TagConstants.toString(TagConstants.LBLPOS) :
                TagConstants.toString(TagConstants.LBLNEG)));
      write(o, " ");
      write(o, le.label.toString());
      write(o, ' ');
      self.print(o, ind, le.expr);
      write(o, ")");
      break;
    }
      
    case TagConstants.LOCKSETEXPR:
      write(o, TagConstants.toString(TagConstants.LS));
      break;
      
    case TagConstants.ELEMSNONNULL:
    case TagConstants.ELEMTYPE:
    case TagConstants.FRESH:
    case TagConstants.MAX:
    case TagConstants.NOWARN_OP:
    case TagConstants.PRE:
    case TagConstants.SPACE:
    case TagConstants.TYPEOF:
    case TagConstants.WACK_BIGINT_MATH: 
    case TagConstants.WACK_DURATION:
    case TagConstants.WACK_JAVA_MATH:
    case TagConstants.WACK_NOWARN:
    case TagConstants.WACK_SAFE_MATH:
    case TagConstants.WACK_WORKING_SPACE:
    case TagConstants.WARN:
    case TagConstants.WARN_OP: {
      write(o, TagConstants.toString(tag));
      self.print(o, ind, ((NaryExpr)e).exprs);
      break;
    }
      
    case TagConstants.NOT_MODIFIED: 
      write(o, TagConstants.toString(tag));
      write(o, '(');
      self.print(o, ind, ((NotModifiedExpr)e).expr);
      write(o, ')');
      break;
      
    case TagConstants.DTTFSA: {
      ExprVec es = ((NaryExpr)e).exprs;
      write(o, TagConstants.toString(tag));
      write(o, '(');
      self.print(o, ((TypeExpr)es.elementAt(0)).type);
      for (int i = 1; i < es.size(); i++) {
        write(o, ", ");
        self.print(o, ind, es.elementAt(i));
      }
      write(o, ')');
      break;
    }
      
    case TagConstants.NUM_OF:{
      NumericalQuantifiedExpr qe = (NumericalQuantifiedExpr)e;
      write(o, "(");
      write(o, TagConstants.toString(tag));
      write(o, " ");
      String prefix = "";
      for( int i=0; i<qe.vars.size(); i++) {
        GenericVarDecl decl = qe.vars.elementAt(i);
        write(o, prefix );
        if (i == 0) self.print(o, decl.type);
        write(o, ' ');
        if (escjava.Main.options().nvu)
          write(o, decl.id.toString());
        else
          write(o, escjava.translate.UniqName.variable(decl));
        prefix = ", ";
      }
      write(o, "; ");
      self.print(o, ind, qe.expr);
      write(o, ')');
      break;
    }
      
    case TagConstants.SUM:
    case TagConstants.PRODUCT:
    case TagConstants.MIN:
    case TagConstants.MAXQUANT:{
      GeneralizedQuantifiedExpr qe = (GeneralizedQuantifiedExpr)e;
      write(o, "(");
      write(o, TagConstants.toString(tag));
      write(o, " ");
      String prefix = "";
      for( int i=0; i<qe.vars.size(); i++) {
        GenericVarDecl decl = qe.vars.elementAt(i);
        write(o, prefix );
        if (i == 0) self.print(o, decl.type);
        write(o, ' ');
        if (escjava.Main.options().nvu)
          write(o, decl.id.toString());
        else
          write(o, escjava.translate.UniqName.variable(decl));
        prefix = ", ";
      }
      write(o, "; ");
      self.print(o, ind, qe.expr);
      write(o, "; ");
      self.print(o, ind, qe.rangeExpr);
      write(o, ')');
      break;
    }
    case TagConstants.FORALL:
    case TagConstants.EXISTS: {
      QuantifiedExpr qe = (QuantifiedExpr)e;
      write(o, "(");
      write(o, TagConstants.toString(tag));
      write(o, " ");
      String prefix = "";
      for( int i=0; i<qe.vars.size(); i++) {
        GenericVarDecl decl = qe.vars.elementAt(i);
        write(o, prefix );
        if (i == 0) self.print(o, decl.type);
        write(o, ' ');
        if (escjava.Main.options().nvu)
          write(o, decl.id.toString());
        else
          write(o, escjava.translate.UniqName.variable(decl));
        prefix = ", ";
      }
      write(o, "; ");
      self.print(o, ind, qe.expr);
      write(o, ')');
      break;
    }
      
    case TagConstants.RESEXPR:
      write(o, TagConstants.toString(TagConstants.RES));
      break;
      
    case TagConstants.BOOLEANLIT: {
      String comment = (String)EscPragmaParser.informalPredicateDecoration.get(e);
      if (comment != null) {
        LiteralExpr le = (LiteralExpr)e;
        Boolean b = (Boolean)le.value;
        Assert.notFalse(b.booleanValue() == true);
          
        write(o, "(*");
        write(o, comment);
        write(o, "*)");
      } else {
        super.print(o, ind, e);
      }
      break;
    }
      
    case TagConstants.SUBSTEXPR: {
      SubstExpr subst = (SubstExpr)e;
      write(o, "[subst ");
      self.print(o, ind, subst.val);
      write(o, " for ");
      if (escjava.Main.options().nvu)
        write(o, subst.var.id.toString());
      else
        write(o, escjava.translate.UniqName.variable(subst.var));
      write(o, " in ");
      self.print(o, ind, subst.target);
      write(o, "]");
      break;
    }
      
    case TagConstants.TYPEEXPR:
      write(o, TagConstants.toString(TagConstants.TYPE));
      write(o, "(");
      self.print(o, ((TypeExpr)e).type);
      write(o, ")");
      break;
      
    case TagConstants.ARRAYRANGEREFEXPR: {
      ArrayRangeRefExpr we = (ArrayRangeRefExpr)e;
      print( o, ind, we.array );
      write(o, "[");
      if (we.lowIndex == null && we.highIndex == null) {
        write(o, "*");
      } else {
        print(o, ind, we.lowIndex);
        write(o,"..");
        print(o, ind, we.highIndex);
      }
      write(o, "]");
      break;
    }
      
    case TagConstants.WILDREFEXPR: {
      WildRefExpr we = (WildRefExpr)e;
      print( o, ind, we.od );
      // The ObjectDesignator prints the '.'
      write(o, "*");
      break;
    }
      
    case TagConstants.GUARDEXPR: {
      GuardExpr ge = (GuardExpr)e;
      write ( o, "(GUARD ");
      write ( o, escjava.translate.UniqName.locToSuffix(ge.locPragmaDecl) + " ");
      print( o, ind, ge.expr );
      write(o, ")");
      break;
    }
      
    case TagConstants.ALLOCLT:
    case TagConstants.ALLOCLE:
    case TagConstants.ANYEQ:
    case TagConstants.ANYNE:
    case TagConstants.ARRAYLENGTH:
    case TagConstants.ARRAYFRESH:
    case TagConstants.ARRAYMAKE:
    case TagConstants.ARRAYSHAPEMORE:
    case TagConstants.ARRAYSHAPEONE:
    case TagConstants.ASELEMS:
    case TagConstants.ASFIELD:
    case TagConstants.ASLOCKSET:
    case TagConstants.BOOLAND:
    case TagConstants.BOOLANDX:
    case TagConstants.BOOLEQ:
    case TagConstants.BOOLIMPLIES:
    case TagConstants.BOOLNE:
    case TagConstants.BOOLNOT:
    case TagConstants.BOOLOR:
    case TagConstants.CAST:
    case TagConstants.CLASSLITERALFUNC:
    case TagConstants.CONDITIONAL:
    case TagConstants.ECLOSEDTIME:
    case TagConstants.FCLOSEDTIME:
    case TagConstants.FLOATINGADD:
    case TagConstants.FLOATINGDIV:
    case TagConstants.FLOATINGEQ:
    case TagConstants.FLOATINGGE:
    case TagConstants.FLOATINGGT:
    case TagConstants.FLOATINGLE:
    case TagConstants.FLOATINGLT:
    case TagConstants.FLOATINGMOD:
    case TagConstants.FLOATINGMUL:
    case TagConstants.FLOATINGNE:
    case TagConstants.FLOATINGNEG:
    case TagConstants.FLOATINGSUB:
    case TagConstants.INTEGRALADD:
    case TagConstants.INTEGRALAND:
    case TagConstants.INTEGRALDIV:
    case TagConstants.INTEGRALEQ:
    case TagConstants.INTEGRALGE:
    case TagConstants.INTEGRALGT:
    case TagConstants.INTEGRALLE:
    case TagConstants.INTEGRALLT:
    case TagConstants.INTEGRALMOD:
    case TagConstants.INTEGRALMUL:
    case TagConstants.INTEGRALNE:
    case TagConstants.INTEGRALNEG:
    case TagConstants.INTEGRALNOT:
    case TagConstants.INTEGRALOR:
    case TagConstants.INTSHIFTL:
    case TagConstants.LONGSHIFTL:
    case TagConstants.INTSHIFTR:
    case TagConstants.LONGSHIFTR:
    case TagConstants.INTSHIFTRU:
    case TagConstants.LONGSHIFTRU:
    case TagConstants.INTEGRALSUB:
    case TagConstants.INTEGRALXOR:
    case TagConstants.INTERN:
    case TagConstants.INTERNED:
    case TagConstants.IS:
    case TagConstants.ISALLOCATED:
    case TagConstants.ISNEWARRAY:
    case TagConstants.LOCKLE:
    case TagConstants.LOCKLT:
    case TagConstants.REFEQ:
    case TagConstants.REFNE:
    case TagConstants.SELECT:
    case TagConstants.STORE:
    case TagConstants.STRINGCAT:
    case TagConstants.STRINGCATP:
    case TagConstants.TYPEEQ:
    case TagConstants.TYPENE:
    case TagConstants.TYPELE:
    case TagConstants.UNSET:
    case TagConstants.VALLOCTIME:
      write(o, TagConstants.toString(tag));
      self.print(o, ind, ((NaryExpr)e).exprs);
      break;
      
    case TagConstants.METHODCALL:
      write(o, ((NaryExpr)e).methodName.toString());
      self.print(o, ind, ((NaryExpr)e).exprs);
      break;
      
    case TagConstants.NOTMODIFIEDEXPR:
      write(o, TagConstants.toString(TagConstants.NOT_MODIFIED));
      write(o, "(");
      self.print(o, ind, ((NotModifiedExpr)e).expr);
      write(o, ")");
      break;
      
    case TagConstants.EVERYTHINGEXPR:
      write(o, TagConstants.toString(TagConstants.EVERYTHING));
      break;
    case TagConstants.NOTHINGEXPR:
      write(o, TagConstants.toString(TagConstants.NOTHING));
      break;
    case TagConstants.NOTSPECIFIEDEXPR:
      write(o, TagConstants.toString(TagConstants.NOT_SPECIFIED));
      break;
      
    default:
      Assert.notFalse(tag<=javafe.tc.TagConstants.LAST_TAG,
                      "illegal attempt to pass tag #" + tag + " (" +
                      TagConstants.toString(tag) + ") to javafe");
      super.print(o, ind, e);
      break;
    }
  }
  
  public void print(/*@non_null*/ OutputStream o, /*@ non_null */ Type t) {
    switch ( t.getTag()) {
    case TagConstants.ANY:
      write(o, "anytype" );
      break;
      
    case TagConstants.TYPECODE:
    case TagConstants.LOCKSET:
    case TagConstants.OBJECTSET:
      write(o, TagConstants.toString(t.getTag()) );
      break;
      
    case TagConstants.BIGINTTYPE:
      write(o, "bigint");
      break;
      
    case TagConstants.REALTYPE:
      write(o, "real");
      break;
      
    default:
      super.print( o, t );
    }
  }
  
  /* (non-Javadoc)
   * @see javafe.ast.PrettyPrint#toString(int)
   */
  public /*@ non_null @*/ String toString(int tag) {
    // Best version available in the back end:
    return escjava.ast.TagConstants.toString(tag);
  }
  
}
