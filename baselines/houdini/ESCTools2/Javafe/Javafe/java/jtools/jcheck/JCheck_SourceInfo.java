package jtools.jcheck;


import javafe.filespace.*;


/**
 ** Information about each unit in the sources map.<p>
 **
 ** sourceNode: the file/node in the Java environment that contains
 **		    the source code for this unit.<p>
 **
 ** isInterface: true iff this unit is an interface rather than a class.<p>
 **
 ** isPublic:    true iff this unit is public.<p>
 **/

public class JCheck_SourceInfo {

    public Tree sourceNode;
    public boolean isInterface;
    public boolean isPublic;

    public JCheck_SourceInfo(Tree sourceNode, boolean isInterface,
				boolean isPublic) {
	this.sourceNode = sourceNode;
	this.isInterface = isInterface;
	this.isPublic = isPublic;
    }

}
