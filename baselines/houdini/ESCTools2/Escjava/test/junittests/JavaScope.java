// Tests some scoping issues that Javafe got wrong

public class JavaScope {

	int a;
	int b;
	int d;

	void m() {
		
		a = 0;
		int c;

		class JSB {

			void m(int b) {

				int c = 0;
				int d = a;
				int b = 0; // ERROR
			}
		}

		JSA j = new JSA() {
			public int mt(int c) {
				return a;
			}
		};

	}
}

class JSA {

	public int mt() {return 0;}
}
