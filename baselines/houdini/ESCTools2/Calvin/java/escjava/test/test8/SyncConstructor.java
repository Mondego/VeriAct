/* Copyright Hewlett-Packard, 2002 */

class SyncConstructor {

  /*@ non_null */ Object mu = new Object();
  static /*@ non_null */ Object nu = new Object();

  /*@ monitored */ int x;
  /*@ monitored_by mu; */ int y;
  /*@ monitored_by nu; */ static int g;

  // ----------  WRITE ACCESS  ----------

  SyncConstructor(/*@ non_null */ SyncConstructor that) {
    x = 5;  // no race, because accessor is this
    that.x = 5;  // race, since this != that

    SyncConstructor o = this;
    o.y = 6;  // no race
    o = that;
    o.y = 5;  // race

    g = 5;  // race, because the special rule applies only to instance fields
  }

  SyncConstructor() {
    this.g = 5;  // race (the "this" is irrelevant to static fields)
  }

  void m(/*@ non_null */ SyncConstructor that) {
    x = 5;       // race
    that.x = 5;  // race

    SyncConstructor o = this;
    o.y = 6;  // race
    o = that;
    o.y = 5;  // race

    g = 5;  // race
  }

  // ----------  READ ACCESS  ----------

  SyncConstructor(/*@ non_null */ SyncConstructor that, int q) {
    int k;
    k = x;  // no race, because accessor is this
    k = that.x;  // race, since this != that

    SyncConstructor o = this;
    k = o.y+1;  // no race
    o = that;
    k = o.y;  // race

    k = g;  // race, because the special rule applies only to instance fields
  }

  SyncConstructor(int q) {
    int k;
    k = this.g;  // race (the "this" is irrelevant to static fields)
  }

  void m(/*@ non_null */ SyncConstructor that, int q) {
    int k;
    k = x;       // race
    k = that.x;  // race

    SyncConstructor o = this;
    k = o.y+1;  // race
    o = that;
    k = o.y;  // race

    k = g;  // race
  }
}
