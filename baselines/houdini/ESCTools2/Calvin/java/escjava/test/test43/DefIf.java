/* Copyright Hewlett-Packard, 2002 */

class DefIf {
  int f;
  int h /*@ readable_if f == 2 */;
  
  void m0() {
    int x = 0;
    if (this.f == 2) {
      x = this.h;  // okay
    }
  }
  
  void m1() {
    int x = 0;
    x = this.h;  // illegal access
  }
  
  void m2(DefIf d) {
    int x = 0;
    if (d != null && d.f == 2) {
      x = d.h;  // okay
      Object o = d;
      x = ((DefIf)o).h;  // okay
    }
  }

  void m3(DefIf d) {
    int x = 0;
    if (d != null && this.f == 2) {
      x = d.h;  // illegal access
    }
  }
}

class MonitoredBy {
  /*@ non_null */ Object f = new Object();
  int h /*@ monitored_by f */;

  void m0() {
    int x = 0;
    synchronized (this.f) {
      x = this.h;  // okay
      this.h = x;  // okay
    }
  }
  
  void m1r() {
    int x = 0;
    x = this.h;  // illegal access
  }
  
  void m1w() {
    int x = 0;
    this.h = x;  // illegal access
  }
  
  void m2(/*@ non_null */ MonitoredBy d) {
    int x = 0;
    synchronized (d.f) {
      x = d.h;  // okay
      d.h = x;  // okay
      Object o = d;
      x = ((MonitoredBy)o).h;  // okay
      ((MonitoredBy)o).h = x;  // okay
    }
  }

  void m3r(/*@ non_null */ MonitoredBy d) {
    int x = 0;
    synchronized (this.f) {
      x = d.h;  // illegal access
    }
  }

  void m3w(/*@ non_null */ MonitoredBy d) {
    int x = 0;
    synchronized (this.f) {
      d.h = x;  // illegal access
    }
  }
}

class Monitored {
  /*@ monitored */ int h;

  void m0() {
    int x = 0;
    synchronized (this) {
      x = this.h;  // okay
      this.h = x;  // okay
    }
  }
  
  void m1r() {
    int x = 0;
    x = this.h;  // illegal access
  }
  
  void m1w() {
    int x = 0;
    this.h = x;  // illegal access
  }
  
  void m2(/*@ non_null */ Monitored d) {
    int x = 0;
    synchronized (d) {
      x = d.h;  // okay
      d.h = x;  // okay
      Object o = d;
      x = ((Monitored)o).h;  // okay
      ((Monitored)o).h = x;  // okay
    }
  }

  void m3r(/*@ non_null */ Monitored d) {
    int x = 0;
    synchronized (this) {
      x = d.h;  // illegal access
    }
  }

  void m3w(/*@ non_null */ Monitored d) {
    int x = 0;
    synchronized (this) {
      d.h = x;  // illegal access
    }
  }
}
