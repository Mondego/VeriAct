/**
 ** This file checks to see how good our error messages are for
 ** inheriting incompatible methods
 **/


interface I1 { void foo() throws java.io.IOException; }
interface I2 { void foo(); }

class I3 implements I1, I2 {}
class I4 implements I2, I1 {}




class Top {
    void foo(int x) throws java.io.IOException {}
}
interface ITop {
    void foo(int x);
}

class Middle extends Top {}

class Bottom extends Middle implements ITop {}



interface OneWay { void foo(int x) throws java.io.IOException; }
interface OtherWay { void foo(int x) throws
                          java.lang.CloneNotSupportedException; }
interface BothWays extends OneWay, OtherWay { }


interface OneWay1 { int foo(int x); }
interface OtherWay1 { char foo(int x); }
interface BothWays1 extends OneWay1, OtherWay1 { }
