/**
 ** Test anonymous classes, part II:
 **/


/*
 * Verify no constructors allowed:
 */

class Anon {
    class Top {
	Top() {}
    }


    void m() {
	Top t7 = new Top() { Top(); };             // parse error
    }
}
