class InstInit {
  int f = 2 + 3;
  
  {
    g = 6;
  }
  
  static {
    g++;
  }
  
  static int g = 5 + 7;

  //@ requires 2 <= y;
  InstInit(int y) {
    x += y;
    //@ assert 50 <= x;
    //@ assert f == 11;
    //@ assert g == 13;
  }
  
  {
    f = f + g;  // 11
    g = f + 2; // 13
  }
  
  int x = 23;

  {
    x++;  // 24
    x += f;  // 35
    x += g;  // 48
  }
  
  InstInit() {
    x += 2;
    //@ assert x == 50;
    //@ assert f == 11;
    //@ assert g == 13;
  }

  InstInit(int z0, int z1) {
    x += z0 + z1;
    //@ assert x == 50;   // fails
    //@ assert f == 11;
    //@ assert g == 13;
    //@ assert z0 == 2;   // fails
    /* from the assumptions about x, z0, and z1 above, z1 must be 0, so the next
       assertion will succeed */
    //@ assert z1 == 0;
  }
}
