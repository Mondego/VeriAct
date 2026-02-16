#!/usr/local/bin/perl
# Copyright (c) 1999, Compaq Computer Corporation
# Change history:
#    2 Apr 1999  rustan     Created

# Output format: one of the following:
#   File line col errorType <description> File placement line col id 'pragma'
#   File line col errorType <reason>
# where
#   File is a filename
#   line is a number (the second "line" may be 0 to indicate a location in
#     a binary file; in that case, the second "col" is also 0)
#   col is a number
#   errorType is an ESC/Java error tag, like Null or Cast
#   description categorizes the suggested annotated entity
#   placement is one of <P, <, <<, <|, or >>, which have the following
#   meanings:
#       <P  place just before type
#       <   place just before type (*)
#       <<  place just before type, or preferably on new previous line (*)
#       <|  place here, or preferably on new previous line
#       >>  place after semi-colon, preferably on new next line
#    (*) but don't act if a comma is between the type and the current position,
#        or if a comma follows the current position before a semicolon or
#        open parenthesis does.
#   pragma is any string
#   reason is any string, explaining why there was no suggestion
# Note, if the suggestion refers to a whole-file location, the second "line"
# will be 0 (and so will "col").

# sets $suggId, $suggFile, $suggLine, and $suggCol from a given NAME
sub setNameInfo {
  my $name = $_[0];

  ##  'ID' at M,N in FILE
  if ($name =~ /^'([^']*)' at (\d*),(\d*) in (\S*)$/) {
    $suggId = $1;
    $suggLine = $2;
    $suggCol = $3;
    $suggFile = $4;

  ##  'ID' in FILE
  } elsif ($name =~ /^'([^']*)' in (\S*)$/) {
    $suggId = $1;
    $suggLine = 0;
    $suggCol = 0;
    $suggFile = $2;

  } else {
    die "unexpected NAME field\n";
  }
}

# sets $suggPlacement from a given DESCRIPTION
sub setPlacement {
  my $desc = $_[0];
  if ($desc =~ /parameter$/) {
    $suggPlacement = "<P";
  } elsif ($desc =~ /field$/) {
    $suggPlacement = "<";
  } elsif ($desc =~ /local variable$/) {
    $suggPlacement = "<";
  } elsif ($desc =~ /method$/) {
    $suggPlacement = "<<";
  } elsif ($desc =~ /method override$/) {
    $suggPlacement = "<<";
  } elsif ($desc =~ /constructor$/) {
    $suggPlacement = "<|";
  } elsif ($desc =~ /invariant$/) {
    $suggPlacement = ">>";
  } else {
    die "unexpected ENTITY field ($desc); ",
        "cannot determine suggested placement\n";
  }
}

sub handleWarning {
  my $sugg = $_[0];

  my $entity;
  my $suggAnn;

  ##  none
  if ($sugg =~ /^none$/) {
    print "<>\n";
    return;

  ##  none <reason>
  } elsif ($sugg =~ /^none <(.*)>$/) {
    print "<$1>\n";
    return;

  ##  perhaps declare ENTITY NAME with 'ANN'
  } elsif ($sugg =~ /^perhaps declare (.*) ('[^']*' .*) with '([^']*)'$/) {
    $entity = $1;
    setNameInfo($2);
    $suggAnn = $3;

  ##  perhaps declare ENTITY NAME with 'ANN' (ALTERNATIVE)
  ##  perhaps declare ENTITY NAME with 'ANN' (ADDITIONAL_INFO)
  } elsif ($sugg =~ /^perhaps declare (.*) ('[^']*' .*) with '([^']*)' \(.*\)$/) {
    $entity = $1;
    setNameInfo($2);
    $suggAnn = $3;

  ##  perhaps declare ENTITY 'ANN' in class T (near NAME)
  } elsif ($sugg =~ /^perhaps declare (.*) '([^']*)' in class .* \(near (.*)\)$/) {
    $entity = $1;
    $suggAnn = $2;
    setNameInfo($3);

  } elsif ($sugg =~ /^(add an initializer or initialize the field in constructor)$/) {
    print "<$1>\n";
    return;

  } else {
    die "unknown suggestion kind: $sugg\n";
  }

  setPlacement($entity);
  print "<$entity> $suggFile $suggPlacement $suggLine $suggCol $suggId ",
        "'$suggAnn'\n"
}

while (<STDIN>) {
  chop;
  if ($_ =~ /^([^:\s]*):(\d*): Warning: .* \((.*)\)$/) {
    $warningFile = $1;
    $warningLine = $2;
    $warningLabel = $3;
  } elsif ($_ =~ /Suggestion \[(\d*),(\d*)\]: (.*)$/) {
    if ($warningLine != $1) {
      die "inconsistent line numbers: $warningLine vs $1";
    }
    print "$warningFile $warningLine $2 $warningLabel ";
    handleWarning($3)
  } else {
    # ignore this line
  }
}
