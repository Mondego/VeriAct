public class JavaScopeD extends JavaScopeDD.II {

	public class I {

		void m() {

			J j = new J();
		}
	}
}

class JavaScopeDD {

	public class II {}

	public class J {}
}
