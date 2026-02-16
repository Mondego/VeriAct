class Error0 {
  int[] a;

  // The following pragma is not syntactically valid, because
  // inside JavaDoc comments, the non-white character that is
  // skipped at the beginning of a line is '*', not '@'.
  
  /**     <esc>modifies a;</esc>
<esc>   ensures \result instanceof Object[] ==>
  @             \nonnullelements((Object[])\result);</esc>
    */
  Object m() {
    return a;
  }
}
