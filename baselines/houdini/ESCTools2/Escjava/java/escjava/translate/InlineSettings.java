/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.translate;

class InlineSettings {
  boolean dontCheckPreconditions;
  boolean dontCheckInlinedBody;
  boolean getSpecForInline; // as opposed to getSpecForBody

  int nextInlineCheckDepth = 0;
  int nextInlineDepthPastCheck = 0;

  InlineSettings(boolean dontCheckPreconditions,
		 boolean dontCheckInlinedBody,
		 boolean getSpecForInline) {
    this.dontCheckPreconditions = dontCheckPreconditions;
    this.dontCheckInlinedBody = dontCheckInlinedBody;
    this.getSpecForInline = getSpecForInline;
  }

  InlineSettings(boolean dontCheckPreconditions,
		 boolean dontCheckInlinedBody,
		 boolean getSpecForInline,
		 int checkDepth, int depthPastCheck) {
    this(dontCheckPreconditions,
	 dontCheckInlinedBody,
	 getSpecForInline);
    this.nextInlineCheckDepth = checkDepth;
    this.nextInlineDepthPastCheck = depthPastCheck;
  }

  InlineSettings(InlineSettings is,
		 int checkDepth, int depthPastCheck) {
    this(is.dontCheckPreconditions,
	 is.dontCheckInlinedBody,
	 is.getSpecForInline);
    this.nextInlineCheckDepth = checkDepth;
    this.nextInlineDepthPastCheck = depthPastCheck;
  }
}
