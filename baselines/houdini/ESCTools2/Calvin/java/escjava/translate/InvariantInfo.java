/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;


import java.util.Hashtable;
import java.util.Enumeration;

import escjava.Main;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;

import escjava.backpred.FindContributors;

import javafe.tc.TypeSig;
import escjava.tc.Types;
import escjava.tc.TypeCheck;

import javafe.util.*;

/**
 ** This class is used by <code>collectInvariants</code> and its callers,
 ** <code>extendSpecForCall</code> and <code>extendSpecForBody</code>.
 **/

class InvariantInfo {
  ExprDeclPragma prag;
  TypeSig U;           // "TypeSig" of class or interface that contains "prag"
  boolean isStatic;    // "true" if pragma declares a static invariant
  LocalVarDecl sdecl;  // "null" if "isStatic"; otherwise, a dummy variable
  VariableAccess s;    // "null" if "isStatic"; otherwise, a variable access
                       // of "sdecl"
  Hashtable map;       // "{this-->s}"
  Expr J;              // translated expression, with substitution
                       // "{this-->s}" if not "isStatic"
  InvariantInfo next;  // for linking "InvariantInfo" nodes
}

