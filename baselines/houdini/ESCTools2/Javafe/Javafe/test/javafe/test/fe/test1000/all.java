class A {

  // javafe doesn't do flow sensitive checking yet
  void f() {
  x: 
    for(;;) break x;
    f();
  }

  // bug in javac, causes compiler crash
  // ok in javafe
  void g() {
  x: 
    for(;;) continue x;
    g();
  }
}
class D {
 void i() {
    { 
    x: 
      i(); 
    } 
  x:
    i();
  }
}
class E {
  void j() {
  x: 
    j(); 
  x:
    j();
  }
}
class G {
  void l() {
  x: 
  y:
    l(); 
  }
}
