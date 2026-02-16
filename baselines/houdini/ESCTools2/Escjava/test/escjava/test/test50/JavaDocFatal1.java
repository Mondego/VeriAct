/**
 ** This file tests parsing of escjava annotations in javadoc comments
 **/

class JavaDocFatal1 {

    void standard() {

      /** <esc>assert true</esc>
	* <esc>notAnEscPragma hello</esc>  <--- error
	* <esc>assume false</esc> */
    }
}
