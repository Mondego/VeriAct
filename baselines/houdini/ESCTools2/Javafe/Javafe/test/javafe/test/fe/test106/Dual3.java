/**
 ** Test ambiguous references to type members (multiple parts)
 **/

class Dual3 extends SuperType implements SuperInterface {
    class T {}         // ok since not a real reference to T
}

class Dual4 extends SuperType implements SuperInterface {
    Object o = this.new T();   // error: ambiguous
}

class SuperType {
    class T { int i; }
}

interface SuperInterface {
    class T { boolean i; }
}
