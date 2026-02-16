// Tests the parsing of the weakly annotations

public class Weakly implements A /*@ weakly */, B, C /*@ weakly */ {

	public void m() {}
}

interface A {}
interface B {}
interface C {}
