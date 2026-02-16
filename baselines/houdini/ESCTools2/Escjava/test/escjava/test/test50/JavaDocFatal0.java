/**
 ** This file tests parsing of escjava annotations in javadoc comments
 **/

class JavaDocFatal0 {

    void standard() {

      /** <esc>notAnEscPragma hello</esc>  <--- error
	* <esc>assert true</esc>
	* <esc>assume false</esc> */
    }
}
