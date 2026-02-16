class Nowarn {
  void m0(int y, int x) {
    switch (y) {
      case 0:
	//@ decreases x;
	while (0 <= x) {  //@ nowarn Decreases
	}
	break;

      case 1:
	//@ decreases x;  //@ nowarn Decreases
	while (0 <= x) {
	}
	break;

      case 2:
	//@ decreases x;
	while (true) {  //@ nowarn DecreasesBound
	  x--;
	}
	break;

      case 3:
	//@ decreases x;  //@ nowarn DecreasesBound
	while (true) {
	  x--;
	}
	break;

      case 4:
	//@ decreases x;  // should give 2 warnings
	while (true) {
	}
	break;

      case 5:
	//@ decreases x;  // should give 1 warning
	while (true) {  //@ nowarn DecreasesBound
	}
	break;

      case 6:
	//@ decreases x;  // should give 1 warning
	while (true) {  //@ nowarn Decreases
	}
	break;

      case 7:
	//@ decreases x;  // should give 0 warnings
	while (true) {  //@ nowarn Decreases, DecreasesBound
	}
	break;

      case 8:
	//@ decreases x;  // should give 0 warnings
	while (true) {  //@ nowarn
	}
	break;
    }
  }

  void m1(int y, int x) {
    switch (y) {
      case 0:
	//@ decreases x;
	while (0 <= x) {  //@ nowarn
	}
	break;

      case 1:
	//@ decreases x;  //@ nowarn
	while (0 <= x) {
	}
	break;

      case 2:
	//@ decreases x;
	while (true) {  //@ nowarn
	  x--;
	}
	break;

      case 3:
	//@ decreases x;  //@ nowarn
	while (true) {
	  x--;
	}
	break;

      case 4:
	//@ decreases x;  // should give 2 warnings
	while (true) {
	}
	break;
    }
  }
}
