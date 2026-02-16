/**
 ** This file tests that FindContributors handles instance
 ** initializer blocks correctly.
 **/

class T {
    int k;
    //@ invariant k>0

    T() {
	k = 1;
    }
}

class U extends T {
    { k = -1; }

    U() {}       // Error
}
