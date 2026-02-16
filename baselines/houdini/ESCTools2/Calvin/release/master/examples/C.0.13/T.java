/* Copyright Hewlett-Packard, 2002 */

class T {
    public static void main(String[] args) {
        m("hel", "lo");
    }

    public static void m(String a, String b) {
        String ab = a + b;
        String ab1 = a + b;
        //@ assert ab == ab1;      // wrong, but passes
        System.out.print(ab + "\n");
        System.out.print(ab1 + "\n");
        System.out.print((ab == ab1) + "\n");
    }
}

