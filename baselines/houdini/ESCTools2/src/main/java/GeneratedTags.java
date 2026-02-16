// -*- mode: java -*-
/* Copyright 2000, 2001, Compaq Computer Corporation */

/* IF THIS IS A JAVA FILE, DO NOT EDIT IT!  

   Most Java files in this directory which are part of the Javafe AST
   are automatically generated using the astgen comment (see
   ESCTools/Javafe/astgen) from the input file 'hierarchy.h'.  If you
   wish to modify AST classes or introduce new ones, modify
   'hierarchy.j.'
 */

package escjava.ast;

import javafe.ast.ASTDecoration;
import javafe.ast.ASTDotVisitor;
import javafe.ast.ASTNode;
import javafe.ast.AmbiguousMethodInvocation;
import javafe.ast.AmbiguousVariableAccess;
import javafe.ast.ArrayInit;
import javafe.ast.ArrayRefExpr;
import javafe.ast.ArrayType;
import javafe.ast.AssertStmt;
import javafe.ast.BinaryExpr;
import javafe.ast.BlockStmt;
import javafe.ast.BranchStmt;
import javafe.ast.BreakStmt;
import javafe.ast.CastExpr;
import javafe.ast.CatchClause;
import javafe.ast.CatchClauseVec;
import javafe.ast.ClassDecl;
import javafe.ast.ClassDeclStmt;
import javafe.ast.ClassLiteral;
import javafe.ast.CompilationUnit;
import javafe.ast.CompoundName;
import javafe.ast.CondExpr;
import javafe.ast.ConstructorDecl;
import javafe.ast.ConstructorInvocation;
import javafe.ast.ContinueStmt;
import javafe.ast.DefaultVisitor;
import javafe.ast.DelegatingPrettyPrint;
import javafe.ast.DoStmt;
import javafe.ast.ErrorType;
import javafe.ast.EvalStmt;
import javafe.ast.Expr;
import javafe.ast.ExprObjectDesignator;
import javafe.ast.ExprVec;
import javafe.ast.FieldAccess;
import javafe.ast.FieldDecl;
import javafe.ast.FieldDeclVec;
import javafe.ast.ForStmt;
import javafe.ast.FormalParaDecl;
import javafe.ast.FormalParaDeclVec;
// import javafe.ast.GeneratedTags;
import javafe.ast.GenericBlockStmt;
import javafe.ast.GenericVarDecl;
import javafe.ast.GenericVarDeclVec;
import javafe.ast.IdPragma;
import javafe.ast.Identifier;
import javafe.ast.IdentifierNode;
import javafe.ast.IdentifierVec;
import javafe.ast.IfStmt;
import javafe.ast.ImportDecl;
import javafe.ast.ImportDeclVec;
import javafe.ast.InitBlock;
import javafe.ast.InstanceOfExpr;
import javafe.ast.InterfaceDecl;
import javafe.ast.JavafePrimitiveType;
import javafe.ast.LabelStmt;
import javafe.ast.LexicalPragma;
import javafe.ast.LexicalPragmaVec;
import javafe.ast.LiteralExpr;
import javafe.ast.LocalVarDecl;
import javafe.ast.LocalVarDeclVec;
import javafe.ast.MethodDecl;
import javafe.ast.MethodDeclVec;
import javafe.ast.MethodInvocation;
import javafe.ast.ModifierPragma;
import javafe.ast.ModifierPragmaVec;
import javafe.ast.Modifiers;
import javafe.ast.Name;
import javafe.ast.NewArrayExpr;
import javafe.ast.NewInstanceExpr;
import javafe.ast.ObjectDesignator;
import javafe.ast.OnDemandImportDecl;
import javafe.ast.OperatorTags;
import javafe.ast.ParenExpr;
import javafe.ast.PrettyPrint;
import javafe.ast.PrimitiveType;
import javafe.ast.ReturnStmt;
import javafe.ast.RoutineDecl;
import javafe.ast.SimpleName;
import javafe.ast.SingleTypeImportDecl;
import javafe.ast.SkipStmt;
import javafe.ast.StandardPrettyPrint;
import javafe.ast.Stmt;
import javafe.ast.StmtPragma;
import javafe.ast.StmtVec;
import javafe.ast.SuperObjectDesignator;
import javafe.ast.SwitchLabel;
import javafe.ast.SwitchStmt;
import javafe.ast.SynchronizeStmt;
// import javafe.ast.TagConstants;
import javafe.ast.ThisExpr;
import javafe.ast.ThrowStmt;
import javafe.ast.TryCatchStmt;
import javafe.ast.TryFinallyStmt;
import javafe.ast.Type;
import javafe.ast.TypeDecl;
import javafe.ast.TypeDeclElem;
import javafe.ast.TypeDeclElemPragma;
import javafe.ast.TypeDeclElemVec;
import javafe.ast.TypeDeclVec;
import javafe.ast.TypeModifierPragma;
import javafe.ast.TypeModifierPragmaVec;
import javafe.ast.TypeName;
import javafe.ast.TypeNameVec;
import javafe.ast.TypeObjectDesignator;
import javafe.ast.UnaryExpr;
import javafe.ast.Util;
import javafe.ast.VarDeclStmt;
import javafe.ast.VarInit;
import javafe.ast.VarInitVec;
import javafe.ast.VariableAccess;
// import javafe.ast.Visitor;
// import javafe.ast.VisitorArgResult;
import javafe.ast.WhileStmt;

import java.util.Hashtable;
import java.util.Set;
import java.util.ArrayList;

import javafe.util.Assert;
import javafe.util.Location;
import escjava.ParsedRoutineSpecs;

// Convention: unless otherwise noted, integer fields named "loc" refer
// to the location of the first character of the syntactic unit

//# TagBase javafe.tc.TagConstants.LAST_TAG + 1
//# VisitorRoot javafe.ast.Visitor

public class GeneratedTags extends javafe.tc.TagConstants {
   static public final int SUBSTEXPR = javafe.tc.TagConstants.LAST_TAG + 1;
   static public final int TYPEEXPR = SUBSTEXPR + 1;
   static public final int LABELEXPR = TYPEEXPR + 1;
   static public final int WILDREFEXPR = LABELEXPR + 1;
   static public final int GUARDEXPR = WILDREFEXPR + 1;
   static public final int RESEXPR = GUARDEXPR + 1;
   static public final int SETCOMPEXPR = RESEXPR + 1;
   static public final int LOCKSETEXPR = SETCOMPEXPR + 1;
   static public final int EVERYTHINGEXPR = LOCKSETEXPR + 1;
   static public final int NOTHINGEXPR = EVERYTHINGEXPR + 1;
   static public final int NOTSPECIFIEDEXPR = NOTHINGEXPR + 1;
   static public final int NOTMODIFIEDEXPR = NOTSPECIFIEDEXPR + 1;
   static public final int ARRAYRANGEREFEXPR = NOTMODIFIEDEXPR + 1;
   static public final int DEFPREDLETEXPR = ARRAYRANGEREFEXPR + 1;
   static public final int DEFPREDAPPLEXPR = DEFPREDLETEXPR + 1;
   static public final int GETSCMD = DEFPREDAPPLEXPR + 1;
   static public final int SUBGETSCMD = GETSCMD + 1;
   static public final int SUBSUBGETSCMD = SUBGETSCMD + 1;
   static public final int RESTOREFROMCMD = SUBSUBGETSCMD + 1;
   static public final int VARINCMD = RESTOREFROMCMD + 1;
   static public final int DYNINSTCMD = VARINCMD + 1;
   static public final int SEQCMD = DYNINSTCMD + 1;
   static public final int DECREASESINFO = SEQCMD + 1;
   static public final int LOOPCMD = DECREASESINFO + 1;
   static public final int CALL = LOOPCMD + 1;
   static public final int MODELDECLPRAGMA = CALL + 1;
   static public final int MODELCONSTRUCTORDECLPRAGMA = MODELDECLPRAGMA + 1;
   static public final int MODELTYPEPRAGMA = MODELCONSTRUCTORDECLPRAGMA + 1;
   static public final int MODELMETHODDECLPRAGMA = MODELTYPEPRAGMA + 1;
   static public final int GHOSTDECLPRAGMA = MODELMETHODDECLPRAGMA + 1;
   static public final int STILLDEFERREDDECLPRAGMA = GHOSTDECLPRAGMA + 1;
   static public final int IDENTIFIERMODIFIERPRAGMA = STILLDEFERREDDECLPRAGMA + 1;
   static public final int SETSTMTPRAGMA = IDENTIFIERMODIFIERPRAGMA + 1;
   static public final int SKOLEMCONSTANTPRAGMA = SETSTMTPRAGMA + 1;
   static public final int MODIFIESGROUPPRAGMA = SKOLEMCONSTANTPRAGMA + 1;
   static public final int REACHMODIFIERPRAGMA = MODIFIESGROUPPRAGMA + 1;
   static public final int NOWARNPRAGMA = REACHMODIFIERPRAGMA + 1;
   static public final int IMPORTPRAGMA = NOWARNPRAGMA + 1;
   static public final int REFINEPRAGMA = IMPORTPRAGMA + 1;
   static public final int SPEC = REFINEPRAGMA + 1;
   static public final int CONDITION = SPEC + 1;
   static public final int DEFPRED = CONDITION + 1;
   static public final int LAST_TAG = DEFPRED;


    static public /*@ non_null @*/ String toString(int tag) {
      switch (tag) {
        case SUBSTEXPR: return "SUBSTEXPR";
        case TYPEEXPR: return "TYPEEXPR";
        case LABELEXPR: return "LABELEXPR";
        case WILDREFEXPR: return "WILDREFEXPR";
        case GUARDEXPR: return "GUARDEXPR";
        case RESEXPR: return "RESEXPR";
        case SETCOMPEXPR: return "SETCOMPEXPR";
        case LOCKSETEXPR: return "LOCKSETEXPR";
        case EVERYTHINGEXPR: return "EVERYTHINGEXPR";
        case NOTHINGEXPR: return "NOTHINGEXPR";
        case NOTSPECIFIEDEXPR: return "NOTSPECIFIEDEXPR";
        case NOTMODIFIEDEXPR: return "NOTMODIFIEDEXPR";
        case ARRAYRANGEREFEXPR: return "ARRAYRANGEREFEXPR";
        case DEFPREDLETEXPR: return "DEFPREDLETEXPR";
        case DEFPREDAPPLEXPR: return "DEFPREDAPPLEXPR";
        case GETSCMD: return "GETSCMD";
        case SUBGETSCMD: return "SUBGETSCMD";
        case SUBSUBGETSCMD: return "SUBSUBGETSCMD";
        case RESTOREFROMCMD: return "RESTOREFROMCMD";
        case VARINCMD: return "VARINCMD";
        case DYNINSTCMD: return "DYNINSTCMD";
        case SEQCMD: return "SEQCMD";
        case DECREASESINFO: return "DECREASESINFO";
        case LOOPCMD: return "LOOPCMD";
        case CALL: return "CALL";
        case MODELDECLPRAGMA: return "MODELDECLPRAGMA";
        case MODELCONSTRUCTORDECLPRAGMA: return "MODELCONSTRUCTORDECLPRAGMA";
        case MODELTYPEPRAGMA: return "MODELTYPEPRAGMA";
        case MODELMETHODDECLPRAGMA: return "MODELMETHODDECLPRAGMA";
        case GHOSTDECLPRAGMA: return "GHOSTDECLPRAGMA";
        case STILLDEFERREDDECLPRAGMA: return "STILLDEFERREDDECLPRAGMA";
        case IDENTIFIERMODIFIERPRAGMA: return "IDENTIFIERMODIFIERPRAGMA";
        case SETSTMTPRAGMA: return "SETSTMTPRAGMA";
        case SKOLEMCONSTANTPRAGMA: return "SKOLEMCONSTANTPRAGMA";
        case MODIFIESGROUPPRAGMA: return "MODIFIESGROUPPRAGMA";
        case REACHMODIFIERPRAGMA: return "REACHMODIFIERPRAGMA";
        case NOWARNPRAGMA: return "NOWARNPRAGMA";
        case IMPORTPRAGMA: return "IMPORTPRAGMA";
        case REFINEPRAGMA: return "REFINEPRAGMA";
        case SPEC: return "SPEC";
        case CONDITION: return "CONDITION";
        case DEFPRED: return "DEFPRED";
        default: return javafe.tc.TagConstants.toString(tag); 
      }
    }
}
