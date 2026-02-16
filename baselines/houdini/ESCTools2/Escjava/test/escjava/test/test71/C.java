class C {
  C() {} //@ nowarn

  boolean b, c, d;

  //@ invariant b ==> c;
  //@ invariant b <== c;
  //@ invariant b <==> c;
  //@ invariant b <=!=> c;

  int a[];
  //@ invariant (\forall int j; a[j] > 0) ==> (\exists int j; a[j] > 0);
  //@ invariant (\forall int j; a[j] > 0) <== (\exists int j; a[j] > 0);
  //@ invariant (\forall int j; a[j] > 0) <==> (\exists int j; a[j] > 0);
  //@ invariant (\forall int j; a[j] > 0) <=!=> (\exists int j; a[j] > 0);

  //@ invariant b ==> c ==> d;
  //@ invariant b <== c <== d;
  //@ invariant b <==> c <==> d;
  //@ invariant b <=!=> c <=!=> d;
  
  //@ invariant b ==> (c <== d);
  //@ invariant (b ==> c) <== d;
  //@ invariant b <== (c ==> d);
  //@ invariant (b <== c) ==> d;
  
  //@ invariant b <==> c <=!=> d;
  //@ invariant b <==> c <=!=> d;
}
