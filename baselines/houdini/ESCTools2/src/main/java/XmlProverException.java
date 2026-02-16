package escjava.vcGeneration.xml;

import java.io.IOException;

/**
 * This Exception class is used to signal to escjava.Main when the XmlProver 
 * class has hit an irrecoverable error.
 * 
 * @author Carl Pulley
 *
 */
public class XmlProverException extends RuntimeException {

    public String stylesheet;
    
    public XmlProverException(String stylesheet) {
        this.stylesheet = stylesheet;
    }
    
}
