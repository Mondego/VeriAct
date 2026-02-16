 package escjava.vcGeneration.xml;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;

import javafe.ast.Expr;
import javafe.util.ErrorSet;

import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.ParserConfigurationException;
import javax.xml.transform.OutputKeys;
import javax.xml.transform.Templates;
import javax.xml.transform.Transformer;
import javax.xml.transform.TransformerConfigurationException;
import javax.xml.transform.TransformerException;
import javax.xml.transform.TransformerFactory;
import javax.xml.transform.dom.DOMSource;
import javax.xml.transform.stream.StreamResult;
import javax.xml.transform.stream.StreamSource;

import org.w3c.dom.Attr;
import org.w3c.dom.Document;
import org.w3c.dom.Element;

import escjava.translate.GC;
import escjava.translate.InitialState;
import escjava.vcGeneration.ProverType;
import escjava.vcGeneration.TNode;
import escjava.vcGeneration.TVisitor;
import escjava.vcGeneration.TypeInfo;
import escjava.vcGeneration.VariableInfo;

/**
 * This class provides a simple mechanism by which new prover plugins to the VC 
 * generator may be implemented.
 * 
 * <p>All one has to do is to provide a stylesheet within the <i>escjava.vcGeneration.xml</i>
 * package or located on the system property search path <i>XMLPROVERPATH</i> for the targeted 
 * prover (eg. <i>simplify.xslt</i>). This stylesheet transforms XML terms, conforming to the 
 * DTD defined by <i>escjava/vcGeneration/xml/xmlprover.dtd</i> (as we are compiling with JDK 1.4 
 * this can not be checked!!), into terms of the targeted prover.
 *  
 * <p>Calling ESC/Java with the nvcg option <i>xml.&lt;prover&gt;</i> enables the target prover 
 * VC terms to be generated (eg. <i>-nvcp xml.simplify</i> will enable VC's for the targeted 
 * prover to be generated using the stylesheet <i>escjava/vcGeneration/xml/simplify.xslt</i> - 
 * supplied by default with ESC/Java).
 * 
 * <p>Using this XML based prover interface should make it easier to build and debug new prover 
 * interfaces to the VC generator.
 * 
 * @author Carl Pulley
 */
public class XmlProver extends ProverType {
    
    /**
     * This field defines the prover based stylesheet that will be used to translate our 
     * generated XML VC terms with calls to {@link XmlProver#getProof}
     */
    private String stylesheet = null;
    
    /**
     * This method only allows the field {@link XmlProver#stylesheet} to be set <b>once</b>.
     * 
     * @param stylesheet target prover's stylesheet
     */
    public void setStyleSheet(String stylesheet) {
        if (this.stylesheet == null) {
            this.stylesheet = stylesheet;
        }
    }
    
    /**
     * Our DTD specifies that term names are of type CDATA. Hence this method is an identity 
     * renaming. The prover based stylesheet may perform a renaming of these term names.
     */
    public String labelRename(String label) {
        return label;
    }
    
    private TXmlVisitor visitor = null;
    
    public TVisitor visitor(Writer out) {
        if (visitor == null) {
            visitor = new TXmlVisitor(out);
        }
        return visitor;
    }

    private Document dom = null;
    private Element node = null;
    
    /**
     * Here we ensure that the resulting XML term is suitable for writing to an output file.
     * 
     * <p>After generating our raw XML term conforming to <i>escjava/vcGeneration/xml/xmlprover.dtd</i>
     * (this can not be checked with JDK 1.4!!), we convert it to a prover VC term using the stylesheet 
     * specified by {@link XmlProver#stylesheet}.
     * 
     * <p>The search rules for the stylesheet to be used are as follows (here we assume that <i>&lt;stylesheet&gt;</i> 
     * is the value of the field {@link XmlProver#stylesheet}):
     * <ul>
     * <li>we first look in <i>escjava/vcGeneration/xml</i> for a file named <i>&lt;stylesheet&gt;.xslt</i></li>
     * <li>next we search a :-separated list of directories specified in the system property 
     * <i>XMLPROVERPATH</i> for a file named <i>&lt;stylesheet&gt;.xslt</i>. <i>XMLPROVERPATH</i> has a default value of '.'</li>
     * </ul>
     */
    public void getProof(Writer out, String proofName, TNode term) {
        try {
            DocumentBuilderFactory dbf = DocumentBuilderFactory.newInstance();
            DocumentBuilder db = dbf.newDocumentBuilder();
            //FIXME Ideally DTD validation is performed here - but we need JDK 1.5 for that!!
            dom = db.newDocument();
            dom.appendChild(dom.createComment("Created by ESC/Java XmlProver"));
            node = dom.createElement("VCTERM");
            Attr nameAttr = dom.createAttribute("name");
            nameAttr.setValue(proofName);
            node.setAttributeNode(nameAttr);
            dom.appendChild(node);
            ((TXmlVisitor) visitor(out)).setDocumentNode(dom, node);
            generateDeclarations(out, term);
            generateTerm(out, term);
            dom.normalize();
            // Finally, we write our XML data structure to the output stream (transforming it if necessary)
            TransformerFactory tf = TransformerFactory.newInstance();
            Transformer tr = null;
            if (stylesheet == null) {
                // No stylesheet present, so simply output our XML to the output stream
                tr = tf.newTransformer();
                tr.setOutputProperty(OutputKeys.INDENT, "yes");
                tr.setOutputProperty(OutputKeys.METHOD, "xml");
                tr.setOutputProperty(
                        "{http://xml.apache.org/xslt}indent-amount", "2");
            } else {
                // We have a stylesheet, so transform our XML data and write result as text to the output stream.
                // Preference is given to stylesheets located within escjava/vcGeneration/xml.
                // Should the stylesheet not be found in that location, then directories specificed within the 
                // system property XMLPROVERPATH are searched.
                // If XMLPROVERPATH is not defined, then it defaults to the current working directory (ie. the 
                // directory within which ESC/Java has been invoked).
                InputStream src = getClass().getResourceAsStream(stylesheet + ".xslt");
                if (src == null) {
                    // The stylesheet is not located within the jar file at escjava/vcGeneration/xml, so we 
                    // first read in the XMLPROVERPATH system property. If this is not defined, then we 
                    // provide a default value of '.'
                    File fileSrc = null;
                    String xmlproverpath = System.getProperty("XMLPROVERPATH",
                            ".");
                    // The following code guards against XMLPROVERPATH being defined as an empty string
                    if (xmlproverpath.equals("")) {
                        xmlproverpath = ".";
                    }
                    // Directories within the system property XMLPROVERPATH are now searched in order to 
                    // locate the stylesheet.
                    String[] path = xmlproverpath.split(":");
                    for (int index = 0; index < path.length; index++) {
                        fileSrc = new File(path[index] + "/" + stylesheet
                                + ".xslt");
                        if (fileSrc.exists() && fileSrc.isFile()
                                && fileSrc.canRead()) {
                    		try {
                    			src = new FileInputStream(fileSrc);
                    		} catch (IOException e) {
                    			ErrorSet.fatal("internal error: " + e.getMessage());
                    		}
                            break;
                        }
                    }
                    if (src == null) {
                        // We didn't find a stylesheet within the XMLPROVERPATH directories, so register an error by raising an exception
                        throw new XmlProverException(stylesheet);
                    } else {
                        System.out.println("[XmlProver: using "
                                + fileSrc.toString() + "]");
                    }
                } else {
                    System.out
                            .println("[XmlProver: using escjava/vcGeneration/xml/"
                                    + stylesheet + ".xslt]");
                }
                Templates transformation = tf
                        .newTemplates(new StreamSource(src));
                tr = transformation.newTransformer();
                //tr.setOutputProperty(OutputKeys.METHOD, "text");
                //tr.setOutputProperty(OutputKeys.STANDALONE, "yes");
                //tr.setOutputProperty(OutputKeys.OMIT_XML_DECLARATION, "yes");
            }
            tr.transform(new DOMSource(dom), new StreamResult(out));
        } catch (TransformerConfigurationException exn) {
            System.out.println("XmlProver: " + exn);
        } catch (TransformerException exn) {
            System.out.println("XmlProver: " + exn);
        } catch (ParserConfigurationException exn) {
            System.out.println("XmlProver: " + exn);
        }
    }

    /**
     * Since our XML mapping of variable names is the identity one, we may simply 
     * return the old variable name here.
     */
    public String getVariableInfo(VariableInfo caller) {
        return caller.old;
    }
    
    /**
     * Since our XML mapping of type names is the identity one, we may simply
     * return the old type name here.
     */
    public String getTypeInfo(TypeInfo caller) {
        return caller.old;
    }

    /**
     * Initialization simply defines an identity map on the known types and names.
     */
    public void init() {
        // Predefined types
        TNode._Reference = TNode.addType("%Reference", "%Reference");
        TNode._Time = TNode.addType("%Time", "%Time");
        TNode._Type = TNode.addType("%Type", "%Type");
        TNode._boolean = TNode.addType("boolean", "boolean");
        TNode._char = TNode.addType("char", "char");
        TNode._DOUBLETYPE = TNode.addType("DOUBLETYPE", "DOUBLETYPE");
        TNode._double = TNode.addType("double", "double");
        TNode._Field = TNode.addType("%Field", "%Field");
        TNode._INTTYPE = TNode.addType("INTTYPE", "INTTYPE");
        TNode._integer = TNode.addType("integer", "integer");
        TNode._float = TNode.addType("float", "float");
        TNode._Path = TNode.addType("%Path", "%Path");
        //TNode._String = addType("String", "String"); FIXME, does this type appears in original proof ?

        // Predefined variables name
        // variables used by the old proof system and that we still need
        TNode.addName("ecReturn", "%Path", "ecReturn");
        TNode.addName("ecThrow", "%Path", "ecThrow");
        TNode.addName("XRES", "%Reference", "XRES");        
    }
    
    /**
     * At this stage, we do not know if the target prover is typed or not. Hence, we add the typing 
     * predicate to the VC term. The prover based stylesheet may eliminate this typing predicate.
     */
    public Expr addTypeInfo(InitialState initState, Expr tree) {
        tree = GC.implies(initState.getInitialState(), tree);
        return tree;
    }
    
    /**
     * All rewriting is handled by the prover based stylesheet, so we do nothing here.
     */
    public TNode rewrite(TNode tree) {
        return tree;
    }

	public void generateDeclarations(/*@non_null*/Writer s, HashMap variableNames) {
        Set keySet = variableNames.keySet();
        Iterator iter = keySet.iterator();
        String keyTemp = null;

        while (iter.hasNext()) {
            keyTemp = (String) iter.next();
            VariableInfo viTemp = (VariableInfo) variableNames.get(keyTemp);
            Element decln = dom.createElement("DECLN");
            if (viTemp != null) {
                Attr nameAttr = dom.createAttribute("name");
                nameAttr.setValue(viTemp.getVariableInfo());
                decln.setAttributeNode(nameAttr);
                if (viTemp.type != null) {
                    Attr typeAttr = dom.createAttribute("type");
                    typeAttr.setValue(viTemp.type.getTypeInfo());
                    decln.setAttributeNode(typeAttr);
                }
            }
            node.appendChild(decln);
        }
    }
    
}
