// Tests that modifiers on axiom statements are not allowed

public class AxiomAccess {

	//@ axiom (\forall int i; i>0; -i < 0);
	//@ static axiom (\forall int i; i>0; -i < 0);
	//@ public axiom (\forall int i; i>0; -i < 0);
	//@ protected axiom (\forall int i; i>0; -i < 0);
	//@ private axiom (\forall int i; i>0; -i < 0);

}
