/**
 ** This file tests parsing of escjava annotations in javadoc comments
 **/

class JavaDoc {

    void standard() {

      /** <esc>assert true</esc> */

      /**<esc>assert true</esc>*/
      /**anyoldjunk<esc>assert true</esc>canbehere*/
      /**<escesc><esc>assert true</esc>*/

      //**<esc></esc>*/              //*<esc></esc>
      /** no esc comment */

      //* <esc>assert true         // no error, because not a JavaDoc comment
      /** <esc>assert true   */    // error
      /** <esc>assert true         // error */
      /** <esc>assert true         // not an error  </esc> */

      /** <esc>assert true</esc>
	* <esc>assume true</esc>
	* <esc>assume false</esc> @ @ @ */
    }
}
