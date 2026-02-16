/**
 ** Test ambiguous references to type members (multiple parts)
 **/

class Dual extends SuperType implements SuperInterface {
    // Straight reference:
    int j = T.i;                 // error: ambiguous
}

class SuperType {
    class T { int i; }
}

interface SuperInterface {
    class T { boolean i; }
}
