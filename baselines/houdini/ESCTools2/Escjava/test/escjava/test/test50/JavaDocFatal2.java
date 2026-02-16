/**
 ** This file tests parsing of escjava annotations in javadoc comments
 **/

class JavaDocFatal2 {

    void standard() {

      /** <esc>assert true</esc>
	* <esc>assume false</esc>
	* <esc>notAnEscPragma hello</esc>  <--- error */
    }
}
