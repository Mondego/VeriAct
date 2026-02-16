package foo.bar;

class B {}

interface Neutral1 {}
interface Neutral2 {}

interface A extends Neutral1, B, Neutral2 {}		// error
