#include <string>
using namespace std;

// JNI wrapper for cvc3 library

#ifndef JNIW_HEADER
#define JNIW_HEADER

#ifdef JNIW_C 
extern "C" 
#endif 
int JNIW_start_solver();

#ifdef JNIW_C 
extern "C" 
#endif
int JNIW_stop_solver();

#ifdef JNIW_C 
extern "C" 
#endif
int JNIW_decl_type(string);

#ifdef JNIW_C 
extern "C" 
#endif
int JNIW_decl_preds(string);

#ifdef JNIW_C 
extern "C" 
#endif
int JNIW_decl_funs(string);

#ifdef JNIW_C 
extern "C" 
#endif
int JNIW_undo(int);

#ifdef JNIW_C 
extern "C" 
#endif
int JNIW_assert(string);

#ifdef JNIW_C 
extern "C" 
#endif
string JNIW_query(string);

#ifdef JNIW_C 
extern "C" 
#endif
int JNIW_set_flags(string);


#endif

