package escjava;
import java.util.ArrayList;
import javafe.ast.ModifierPragma;
import javafe.ast.ModifierPragmaVec;

public class ParsedRoutineSpecs {

    public ModifierPragma initialAlso = null; // null if none
    public ArrayList specs = new ArrayList(5);
    public ArrayList impliesThat = new ArrayList(1);
    public ArrayList examples = new ArrayList(1);
    public ModifierPragmaVec modifiers = ModifierPragmaVec.make();

}
	
