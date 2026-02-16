/**
 ** Test ambiguous references to type members (multiple parts)
 **/

class Dual4 implements SuperInterface1, SuperInterface2 {
    // Straight reference:
    int j = T.i;                 // error: ambiguous
}

interface SuperInterface1 {
    class T { boolean i; }
}

interface SuperInterface2 {
    class T { boolean i; }
}
