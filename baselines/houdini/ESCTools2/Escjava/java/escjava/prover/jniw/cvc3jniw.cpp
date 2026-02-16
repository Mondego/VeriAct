// JNI wrapper for cvc library

// Note: segments of this code are copied/adapted from the cvcl sources.

#include <jni.h>
#include "cvc3jniw.h"
#include "Cvc3Wrapper.h"

#include "vc.h"
#include "compat_hash_map.h"
#include "command_line_flags.h"
//#include "debug.h"
#include <stdio.h>

//#include "exception.h"

#include "eval_exception.h"
#include "parser.h"

#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <utility>


// #define JNIW_DEBUG

using namespace std;
using namespace CVC3;

ValidityChecker *JNIW_vc = NULL; // CVC instance
CLFlags JNIW_flags;
bool JNIW_have_flags = false;

enum status {TRUE,FALSE,UNDET};

//Maps of predicate Exprs to their status
map<Expr,status> JNIW_predMap;
//A LIFO queue of predicates as they appear (and their scope)
vector< pair<Expr,int> > JNIW_predList;
typedef vector< pair<Expr,int> >::iterator JNIW_predListIterator;

int JNIW_scope = 0;
int JNIW_scope_base = 0;


//*******************************************************************
// collect all atomic formulas in specified Expr into predMap and IDPredMap
// This requires that the vc be instantiated before calling (JNIW_start_solver)
void JNIW_getAtomicFormulas(Expr e) {
#ifdef JNIW_DEBUG
  cout << "JNIW_getAtomicFormulas()" << endl;
  flush (cout);
#endif
  if (e.isPropAtom()) {
#ifdef JNIW_DEBUG
  cout << "ATOMIC: " << e.toString() <<endl;
  flush (cout);
#endif
    // We could also register the atoms in question
//      vc->registerAtom(e);
    if (JNIW_predMap.find(e) == JNIW_predMap.end()) 
    {
      JNIW_predMap[e] = UNDET;
      JNIW_predList.insert(JNIW_predList.begin(),make_pair(e,JNIW_scope));
    }
  } else {
#ifdef JNIW_DEBUG
  cout << "NOT ATOMIC: " << e.toString() <<endl;
  flush (cout);
#endif
    vector<Expr> kids = e.getKids();
    vector<Expr>::iterator it;
    for (it = kids.begin(); it !=kids.end();it++) {
      JNIW_getAtomicFormulas(*it);
    }
  }
}

//*******************************************************************
// reads a typedef string & returns the associated Type object
// This requires that the vc be instantiated before calling (JNIW_start_solver)
Type JNIW_readType(InputLanguage lang, const string& instring) 
            throw(Exception)
{
#ifdef JNIW_DEBUG
  cout << "JNIW_readType()" << endl;
  flush (cout);
#endif
  TRACE("commands","readType(",instring,") {");
  stringstream s;
  s << instring;
  Parser *parser = new Parser(JNIW_vc,lang,s,false);
  // raise an exception here
  if (parser->done()) throw EvalException("nothing to parse!"); 
  Expr e;
  e = parser->next();

  if(e.getKind() != RAW_LIST || e.arity() == 0 || e[0].getKind() != ID)
    throw EvalException("Invalid command: "+e.toString());
  
  const string& kindName = e[0][0].getString();
  int kind = JNIW_vc->getEM()->getKind(kindName);
  if(kind == NULL_KIND)
    throw EvalException("Unknown command: "+e.toString());
  switch(kind) {
    case TYPEDEF: {
      DebugAssert(e.arity() == 3, "Bad TYPEDEF");
      Expr def(JNIW_vc->parseExpr(e[2]));
      Type t = JNIW_vc->createType(e[1][0].getString(), def);
      TRACE("commands", "evaluateNext: declare new TYPEDEF: ", t, "");
      return t;
    }
    case TYPE: {
      if(e.arity() < 2)
        throw EvalException("Bad TYPE declaration: "+e.toString());
      Expr::iterator i=e.begin(), iend=e.end();
      Type t;
      ++i; // Skip "TYPE" symbol
      for(; i!=iend; ++i) {
        if(i->getKind() != ID)
          throw EvalException("Type name must be an identifier: "+
                              i->toString());
        t = JNIW_vc->createType((*i)[0].getString());
        TRACE("commands", "evaluateNext: declare new TYPEDECL: ", t, "");
        break;
      }
      return t;
    }
    default: 
      throw EvalException("Unknown typedef: "+e.toString());
  }
}
                                                                                
                                                                                
//*******************************************************************
// reads an expr string & returns the associated Expr object
// This requires that the vc be instantiated before calling (JNIW_start_solver)
Expr JNIW_readExpr(InputLanguage lang, const string& instring)
            throw(Exception)
{
#ifdef JNIW_DEBUG
  cout << "JNIW_readExpr()" << endl;
  flush (cout);
#endif
  TRACE("commands","readExpr(",instring,") {");
  stringstream s;
  s << instring;
  Parser *parser = new Parser(JNIW_vc,lang,s,false);
  if (parser->done()) throw EvalException("nothing to parse!"); 
  Expr e;
  e = parser->next();
                                                                                
  if(e.getKind() != RAW_LIST || e.arity() == 0 || e[0].getKind() != ID)
    throw EvalException("Invalid command: "+e.toString());
                                                                                
  const string& kindName = e[0][0].getString();
  int kind = JNIW_vc->getEM()->getKind(kindName);
  if(kind == NULL_KIND)
    throw EvalException("Unknown command: "+e.toString());
  switch(kind) {
    case QUERY:
    case ASSERT: {
      if(e.arity() != 2)
        throw EvalException("ASSERT requires exactly one argument, but is given "
                            +int2string(e.arity()-1)+":\n "+e.toString());
      TRACE_MSG("commands", "** [commands] Asserting formula");
      return JNIW_vc->parseExpr(e[1]);
    }
    default:
      throw EvalException("Unknown expr format: "+e.toString());
  }
}
                                                                                
//*******************************************************************
// reads an funcdef string & returns the associated Op object
// This requires that the vc be instantiated before calling (JNIW_start_solver)
Op JNIW_readFunDef(InputLanguage lang, const string& instring)
            throw(Exception)
{
#ifdef JNIW_DEBUG
  cout << "JNIW_readFunDef()" << endl;
  flush (cout);
#endif
  TRACE("commands","readFunDef(",instring,") {");
  stringstream s;
  s << instring;
  Parser *parser = new Parser(JNIW_vc,lang,s,false);
  if (parser->done()) throw EvalException("nothing to parse!"); 
  Expr e;
  e = parser->next();
                                                                                
  if(e.getKind() != RAW_LIST || e.arity() == 0 || e[0].getKind() != ID)
    throw EvalException("Invalid command: "+e.toString());
                                                                                
  const string& kindName = e[0][0].getString();
  int kind = JNIW_vc->getEM()->getKind(kindName);
  if(kind == NULL_KIND)
    throw EvalException("Unknown command: "+e.toString());
  switch(kind) {
    case CONST: { // (CONST (id_1 ... id_n) type) or (CONST id type)
      if(e.arity() == 3) {
        Type t(JNIW_vc->parseExpr(e[2]));
        vector<Expr> vars;
        if(e[1].getKind() == RAW_LIST)
          vars = e[1].getKids();
        else
          vars.push_back(e[1]);
        for(vector<Expr>::iterator i=vars.begin(), iend=vars.end(); i!=iend; ++i) {
          if((*i).getKind() != ID)
            throw EvalException("Constant name must be an identifier: "
                                +i->toString());
        }
        Op op;
        if (t.isFunction()) {
          for(vector<Expr>::iterator i=vars.begin(), iend=vars.end();
              i!=iend; ++i) {
            op = JNIW_vc->createOp((*i)[0].getString(), t);
            TRACE("commands", "evaluateNext: declare new uninterpreted function: ",
                  op, "");
          }
          return op;
        }
        else 
          throw EvalException("Not a function: "+e.toString());
      } else
        throw EvalException("Wrong number of arguments in CONST: "+e.toString());
    }
                                                                                
    default:
      throw EvalException("Unknown fundef format: "+e.toString());
  }
}
                                                                                
//*******************************************************************
// reads an const def string & processes it
// also works for function defs, if you don't care about the Op
// This requires that the vc be instantiated before calling (JNIW_start_solver)
void JNIW_readConstDef(InputLanguage lang, const string& instring)
            throw(Exception)
{
#ifdef JNIW_DEBUG
  cout << "JNIW_readConstDef()" << endl;
  flush (cout);
#endif
  TRACE("commands","readConstDef(",instring,") {");
  stringstream s;
  s << instring;
  Parser *parser = new Parser(JNIW_vc,lang,s,false);
  if (parser->done()) throw EvalException("nothing to parse!"); 
  Expr e;
  e = parser->next();
                                                                                
  if(e.getKind() != RAW_LIST || e.arity() == 0 || e[0].getKind() != ID)
    throw EvalException("Invalid command: "+e.toString());
                                                                                
  const string& kindName = e[0][0].getString();
  int kind = JNIW_vc->getEM()->getKind(kindName);
  if(kind == NULL_KIND)
    throw EvalException("Unknown command: "+e.toString());
  switch(kind) {
    case CONST: { // (CONST (id_1 ... id_n) type) or (CONST id type)
      if(e.arity() == 3) {
        Type t(JNIW_vc->parseExpr(e[2]));
        vector<Expr> vars;
        if(e[1].getKind() == RAW_LIST)
          vars = e[1].getKids();
        else
          vars.push_back(e[1]);
        for(vector<Expr>::iterator i=vars.begin(), iend=vars.end(); 
            i!=iend; ++i) 
        {
          if((*i).getKind() != ID)
            throw EvalException("Constant name must be an identifier: "
                                +i->toString());
        }
        if (t.isFunction()) {
          for(vector<Expr>::iterator i=vars.begin(), iend=vars.end();
              i!=iend; ++i) {
            Op op = JNIW_vc->createOp((*i)[0].getString(), t);
            TRACE("commands", 
              "evaluateNext: declare new uninterpreted function: ",op, "");
          }
        }
        else {
          for(vector<Expr>::iterator i=vars.begin(), iend=vars.end();
              i!=iend; ++i) {
            // Add to variable list
            Expr v = JNIW_vc->varExpr((*i)[0].getString(), t);
            TRACE_MSG("commands", "** [commands] Declare new variable");
            TRACE("commands verbose", "evaluateNext: declare new UCONST: ",
                  v, "");
          }
        }
      } else if(e.arity() == 4) { // Constant definition (CONST id type def)
        TRACE_MSG("commands", "** [commands] Define new constant");
        // Set the type for later typechecking
        DebugAssert(e[1].getKind() == ID, "Expected ID kind");
        Type t(JNIW_vc->parseExpr(e[2]));
        Expr def(JNIW_vc->parseExpr(e[3]));
                                                                                
        if(t.isFunction()) {
          JNIW_vc->createOp(e[1][0].getString(), t, def);
          TRACE("commands verbose", "evaluateNext: define new function: ",
                e[1][0].getString(), "");
        } else {
          JNIW_vc->varExpr(e[1][0].getString(), t, def);
          TRACE("commands verbose", "evaluateNext: define new variable: ",
                e[1][0].getString(), "");
        }
      } else
        throw EvalException("Wrong number of arguments in CONST: "
                            +e.toString());
      break;
    }

    default:
      throw EvalException("Unknown constdef format: "+e.toString());
  }
}
//*******************************************************************
// This is the simplified version of the command reader, using the api.
// This produces output to cout (which is not captured here).
// It reads an arbirtary number of commands and processes them.
// This requires that the vc be instantiated before calling (JNIW_start_solver)
void JNIW_readCommands(InputLanguage lang, const string& instring)
{
#ifdef JNIW_DEBUG
  cout << "JNIW_readCommands("<< instring <<")" << endl;
  flush (cout);
#endif
  stringstream s;
  s << instring;
  JNIW_vc->loadFile(s,lang,false);
}

                                                                                

//*******************************************************************
// Interface functions

//*******************************************************************
// begets a new instance of the solver
// returns 0
int JNIW_start_solver()
{
#ifdef JNIW_DEBUG
  cout << "JNIW_start_solver()" << endl;
  flush (cout);
#endif
  if (JNIW_have_flags) {
    JNIW_vc = ValidityChecker::create(JNIW_flags);
  } else {
    JNIW_vc = ValidityChecker::create();
  }
  JNIW_scope_base = JNIW_vc->stackLevel();
  JNIW_scope = JNIW_scope_base;
  JNIW_predMap.clear();
  return 0;
}

//*******************************************************************
// kills the current instance of the solver
// returns 0
int JNIW_stop_solver()
{
#ifdef JNIW_DEBUG
  cout << "JNIW_stop_solver()" << endl;
  flush (cout);
#endif
  JNIW_predMap.clear();
  JNIW_predList.clear();
  delete JNIW_vc;
  JNIW_vc = NULL;
  return 0;
}

//*******************************************************************
// declare a new type (string)
// returns 0 (no problem) or -1 (error)
// this cannot be undone
int JNIW_decl_type(string s)
{
#ifdef JNIW_DEBUG
  cout << "JNIW_decl_type("<< s <<")" << endl;
  flush (cout);
#endif
  try {
    JNIW_readType(JNIW_vc->getEM()->getInputLang(),s);
    return 0;
  } catch (Exception x) {
    cerr << "JNIW_decl_type error: "+x.toString() << endl;
    return -1;
  }
}

//*******************************************************************
// declare a predicate's type (string)
// returns 0 (no problem) or -1 (error)
// currently there is no difference between JNIW_decl_preds and 
// JNIW_decl_funs
// this cannot be undone
int JNIW_decl_preds(string s)
{
#ifdef JNIW_DEBUG
  cout << "JNIW_decl_preds("<< s <<")" << endl;
  flush (cout);
#endif
  try {
    JNIW_readConstDef(JNIW_vc->getEM()->getInputLang(),s);
    return 0;
  } catch (Exception x) {
    cerr << "JNIW_decl_preds error: "+x.toString() << endl;
    return -1;
  }
}

//*******************************************************************
// declare a function's type (string)
// returns 0 (no problem) or -1 (error)
// currently there is no difference between JNIW_decl_preds and 
// JNIW_decl_funs
// this cannot be undone
int JNIW_decl_funs(string s)
{
#ifdef JNIW_DEBUG
  cout << "JNIW_decl_funs("<< s <<")" << endl;
  flush (cout);
#endif
  try {
    JNIW_readConstDef(JNIW_vc->getEM()->getInputLang(),s);
    return 0;
  } catch (Exception x) {
    cerr << "JNIW_decl_funs error: "+x.toString() << endl;
    return -1;
  }
}

//*******************************************************************
// undo up to n (integer) assertions
// returns the current scope number (number of remaining in-place 
// assertions)
// this ensures that the list of predicate references is up to date
int JNIW_undo(int n)
{
#ifdef JNIW_DEBUG
  cout << "JNIW_undo("<< n <<")" << endl;
  flush (cout);
#endif
  if (JNIW_scope-JNIW_scope_base <= n) {
    JNIW_scope = JNIW_scope_base;
    JNIW_predMap.clear();
    JNIW_predList.clear();
  } else if (n > 0) {
    JNIW_scope -=n;
    // clean up the local cache of predicate Exprs
    JNIW_predListIterator it = JNIW_predList.begin();
    while (it != JNIW_predList.end() && (it->second > JNIW_scope)) {
      JNIW_predMap.erase(it->first);
      it = JNIW_predList.erase(it);
    }
  }
  JNIW_vc->popto(JNIW_scope);
  return JNIW_scope;
}

//*******************************************************************
// assert a formula (string) to the current context
// returns either 0 (no problem) or -1 (error)
int JNIW_assert(string s)
{
#ifdef JNIW_DEBUG
  cout << "JNIW_assert("<< s <<")" << endl;
  flush (cout);
#endif
  try {
    Expr e = JNIW_readExpr(JNIW_vc->getEM()->getInputLang(),s);
    JNIW_getAtomicFormulas(e);
    JNIW_vc->push();
    JNIW_scope++;
    JNIW_vc->assertFormula(e);
    return 0;
  } catch (Exception x) {
    cerr << "JNIW_assert error: "+x.toString() << endl;
    return -1;
  }
}

//*******************************************************************
// Perform a query on a formula (string)
// returns a string:
//   Don't know
//   Invalid: <counterexample>
//   Valid
//   JNIW_query ERROR: <errorstring>
// where <counterexample is a space-delimited list of TRUE literals
// (possibly empty).  Each literal is contained in parentheses;
// negative literals begin with a tilde (~).
// Return example: (a) (~bXOR4) (~p(next,prev)) (succ(x) = pred(y))
// this ensures that the list of predicate references is up to date
string JNIW_query(string s)
{
#ifdef JNIW_DEBUG
  cout << "JNIW_query("<< s <<")" << endl;
  flush (cout);
#endif
  stringstream ss;
  int enterscope = JNIW_scope;
  try {
    JNIW_vc->push();
    JNIW_scope++;
    Expr e = JNIW_readExpr(JNIW_vc->getEM()->getInputLang(),s);
    JNIW_getAtomicFormulas(e);
    QueryResult ret;
    ret = JNIW_vc->query(e);
    if (ret==VALID) {
        ss << "Valid";
    } else if (ret==ABORT) {
      ss << "Abort";
    } else {
      if (ret==INVALID) {
        ss << "Invalid: ";
      } else if (ret== UNKNOWN) {
        ss << "Don't know: ";
      }
        // accumulate the
        if (false) {
/*
          ExprMap<Expr> emap;
          JNIW_vc->getConcreteModel(emap);
          for (ExprMap<Expr>::iterator it = emap.begin(); 
               it != emap.end(); it++) {
            if (it->second == JNIW_vc->trueExpr()) 
            {
              ss << "("<< it->first.toString() << ") ";
            } else if (it ->second == JNIW_vc->falseExpr()) {
              ss << "(~"<< it->first.toString() << ") ";
            } 
          }
*/
        } else {
          Expr tmp;
          for (JNIW_predListIterator it = JNIW_predList.begin(); 
               it != JNIW_predList.end(); it++) 
          {
            // only print out determined preds
            tmp = JNIW_vc->simplify(it->first);
            if (tmp == JNIW_vc->trueExpr()) {
              ss << "("<< it->first.toString() << ") ";
            } else if (tmp == JNIW_vc->falseExpr()) {
              ss << "(~"<< it->first.toString() << ") ";
            }
            // clean out the query atoms as encountered
            if (it->second == JNIW_scope) {
              JNIW_predMap.erase(it->first);
              it = JNIW_predList.erase(it);
            }
          }
        }
    }
    JNIW_vc->pop();
  } catch (Exception x) {
    ss << "JNIW_query ERROR: " << x.toString();
  }
  // reset the internal scope counter
  JNIW_scope = enterscope;
  return ss.str();
}

//*******************************************************************
// if the solver is already running, it resets current flags
// if the solver is not already running, it will store the flags
// specified (an empty string stores the default flags)
// flags are only stored until the next call to JNIW_set_flags
int JNIW_set_flags(string s)
{
#ifdef JNIW_DEBUG
  cout << "JNIW_set_flags("<<s<<")" << endl;
  flush (cout);
#endif
  try {
    vector<string> vs;
    vector<string>::iterator it;
    int index, i2, sz;

    if (JNIW_vc != NULL) {
      JNIW_flags = JNIW_vc->getFlags();
      JNIW_have_flags = false;
#ifdef JNIW_DEBUG
  cout << "got current flags" << endl;
  flush (cout);
#endif
    } else if (s == "") {
      JNIW_have_flags = false;
#ifdef JNIW_DEBUG
  cout << "empty flags" << endl;
  flush (cout);
#endif
      return 0;
    } else {
      // these flags are retrievable
      JNIW_flags = ValidityChecker::createFlags();
      JNIW_have_flags = true;
#ifdef JNIW_DEBUG
  cout << "generate new flags" << endl;
  flush (cout);
#endif
//      return -2;
    }

    // turn the string into a vector of strings...
    index = s.find_first_not_of(' ',0);
    while (index != -1) {
      i2 = s.find_first_of(' ',index);
      if (i2 == -1) {
        sz = s.size() - index;
      } else {
        sz = i2 - index;
      }
      string subs = s.substr(index,sz);
      vs.push_back(subs);
      index = s.find_first_not_of(' ',i2);
    }

    // now parse the strings
    // cribbed from CVCL::main.cpp
    for (it = vs.begin(); it != vs.end(); it++) {
      if (it->at(0) == '+' || it->at(0) == '-') {
        vector<string> names;
        bool val = (it->at(0) == '+');
        size_t n = JNIW_flags.countFlags(it->substr(1,it->size()-1),names);
        if (n==0) {
          throw CLException(*it + " does not match any known option");
        } else if (n>1) {
          ostringstream ss;
          ss << *it << " is ambiguous.  Possible matches are:\n";
          for(size_t i=0,iend=names.size(); i!=iend; ++i) {
            ss << "  " << names[i] << "\n";
          }
          throw CLException(ss.str());
        } else {
          string name = names[0];
          // Single match; process the option
          CLFlagType tp = JNIW_flags[name].getType();
          switch(tp) {
          case CLFLAG_BOOL: JNIW_flags.setFlag(name, val); break;
          case CLFLAG_INT:
            it++;
            if(it == vs.end())
              throw CLException(*it+" (-"+name
                                +") expects an integer argument.");
            // FIXME: test for *it being an integer string
            JNIW_flags.setFlag(name, atoi((*it).c_str()));
            break;
          case CLFLAG_STRING:
            it++;
            if(it == vs.end())
              throw CLException(*it+" (-"+name
                                +") expects a string argument.");
            JNIW_flags.setFlag(name, *it);
            break;
          case CLFLAG_STRVEC:
            it++;
            if(it == vs.end())
              throw CLException(*it+" (-"+name
                                +") expects a string argument.");
            JNIW_flags.setFlag(name, pair<string,bool>(*it,val));
            break;
          default:
            DebugAssert(false, "parse_args: Bad flag type: "+int2string(tp));
          }
        }
      } else {
        throw CLException("Unknown flag: "+(*it));
      }
    }

    if (JNIW_vc != NULL) {
      JNIW_vc->reprocessFlags();
#ifdef JNIW_DEBUG
  cout << "reprocessing flags" << endl;
  flush (cout);
#endif
    } 

    return 0;
  } catch (Exception x) {
    cerr << "JNIW_set_flags error: "+x.toString() << endl;
    return -1;
  }

}


//*******************************************************************
extern "C" {
JNIEXPORT jint JNICALL Java_escjava_prover_jniw_Cvc3Wrapper_natStartSolver
  (JNIEnv *env, jobject jo)
{
  return (jint)JNIW_start_solver();
}

JNIEXPORT jint JNICALL Java_escjava_prover_jniw_Cvc3Wrapper_natStopSolver
  (JNIEnv *env, jobject jo)
{
  return (jint)JNIW_stop_solver();
}

JNIEXPORT jint JNICALL Java_escjava_prover_jniw_Cvc3Wrapper_natDeclType
  (JNIEnv *env, jobject jo, jstring js)
{
  const char *s = (env)->GetStringUTFChars(js,NULL);
  jint r = (jint)JNIW_decl_type(s);
  (env)->ReleaseStringUTFChars(js,s);
  return r;
}

JNIEXPORT jint JNICALL Java_escjava_prover_jniw_Cvc3Wrapper_natDeclPreds
  (JNIEnv *env, jobject jo, jstring js)
{
  const char *s = (env)->GetStringUTFChars(js,NULL);
  jint r = (jint)JNIW_decl_preds(s);
  (env)->ReleaseStringUTFChars(js,s);
  return r;
}

JNIEXPORT jint JNICALL Java_escjava_prover_jniw_Cvc3Wrapper_natDeclFuns
  (JNIEnv *env, jobject jo, jstring js)
{
  const char *s = (env)->GetStringUTFChars(js,NULL);
  jint r = (jint)JNIW_decl_funs(s);
  (env)->ReleaseStringUTFChars(js,s);
  return r;
}

JNIEXPORT jint JNICALL Java_escjava_prover_jniw_Cvc3Wrapper_natAssert
  (JNIEnv *env, jobject jo, jstring js)
{
  const char *s = (env)->GetStringUTFChars(js,NULL);
  jint r = (jint)JNIW_assert(s);
  (env)->ReleaseStringUTFChars(js,s);
  return r;
}

JNIEXPORT jstring JNICALL Java_escjava_prover_jniw_Cvc3Wrapper_natQuery
  (JNIEnv *env, jobject jo, jstring js)
{
  const char *s = (env)->GetStringUTFChars(js,NULL);
  const char* rs = JNIW_query(s).c_str();
  (env)->ReleaseStringUTFChars(js,s);
  return (env)->NewStringUTF(rs);
}

JNIEXPORT jint JNICALL Java_escjava_prover_jniw_Cvc3Wrapper_natUndo
  (JNIEnv *env, jobject jo, jint jn)
{
  return (jint)JNIW_undo((int)jn);
}

JNIEXPORT jint JNICALL Java_escjava_prover_jniw_Cvc3Wrapper_natSetFlags
  (JNIEnv *env, jobject jo, jstring js)
{
  const char *s = (env)->GetStringUTFChars(js,NULL);
  jint r = (jint)JNIW_set_flags(s);
  (env)->ReleaseStringUTFChars(js,s);
  return r;
}

}
