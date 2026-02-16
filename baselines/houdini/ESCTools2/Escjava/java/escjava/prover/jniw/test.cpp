using namespace std;

#include "cvc3jniw.h"
#include <iostream>

int main() {

  if (JNIW_set_flags("-tcc")) {cout << "ERROR o" << endl;}
  if (JNIW_start_solver()) {cout << "ERROR 1" << endl;}
  // note the following only recognizes one "command"!
  if (JNIW_decl_funs("x:INT;y:INT;z:INT;")) {cout << "ERROR 2" << endl;}
  if (JNIW_decl_funs("y,z:INT;")) {cout << "ERROR 2" << endl;}
  if (JNIW_decl_funs("a,b,c:BOOLEAN;")) {cout << "ERROR 2" << endl;}
  if (JNIW_assert("ASSERT x=y;")) {cout << "ERROR 3" << endl;}
  if (JNIW_assert("ASSERT z=y;")) {cout << "ERROR 4" << endl;}
  if (JNIW_assert("ASSERT a AND b AND NOT c;")) {cout << "ERROR 4" << endl;}
  

  string s = JNIW_query("QUERY NOT x=z;");
  cout << s << endl;

  if (JNIW_stop_solver()) {cout << "ERROR 9" << endl;}

  return 0;
}
