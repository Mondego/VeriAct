#!/usr/local/bin/perl
# Copyright (c) 1999, Compaq Computer Corporation
# Change history:
#   15 Jul 1999  rustan & flanagan  Created
#   31 Aug 1999  rustan & flanagan  Don't remove annotations, but comment them out
#   22 Nov 199   rustan & flanagan  New command-line format

# Input format:
#   Each line has the form:
#     filename line col warning
#   where within each 'filename', 'col' is in decreasing order
#   ('line' may be in any order).  'line' is 1-based whereas 'col'
#   is 0-based.
# Command-line argument:
#   source-filename
# This perl script changes the contents of 'source-filename':  it alters the
# houdini pragma (if any) at location 'line', 'col' for each input line:
#   source-filename line col
# A "houdini pragma" has the form:
#   /*@(houdini:...) ... */
# all fitting on one line, and where the first ellipsis contains no
# close paren and the second ellipsis contains no occurrences of the
# substring "*/".  The 'line', 'col' designates the first character
# of the second ellipsis.  The houdini pragma is changed to
#   /* REMOVED @(houdini:...) ... BECAUSE warning */
# where 'warning' refers to the 'warning' in the input line.
# For each pragma so changed, this perl script outputs the line of the
# input that triggered the change.
#
# This perl script terminates with status of 0, 1, or 2:
#  0  success, output equals input
#  1  success, output differs from input
#  2  some error occurred (for example, 'source-filename' was not found,
#     input is malformed)

sub die2 {
  my $name = $_[0];
  $! = 2;
  die $name;
}


$sourcefilename = $ARGV[0];

$hasSource = 0;
$changedOutput = 0;
while (<STDIN>) {
    chop;
    $inputline = $_;
    ($file,$line,$col,$warning) = split(/ /, $_, 4);

    ($file eq $sourcefilename) || next;

    if (! $hasSource) {
	open(SOURCE, $sourcefilename) || die2 "can't open $sourcefilename";
	@contents = <SOURCE>;
	close SOURCE;
	$hasSource = 1;
    }

    $str = $contents[$line-1];
    chop $str;
    $strLeft = substr($str, 0, $col);
    $strRight = substr($str, $col);

    ($strLeft =~ /(^.*\/\*)(@\(houdini:[^\)]*\) $)/) || next;
    $newLeft = $1 . " REMOVED " . $2;

    ($strRight =~ /(.*?)(\*\/.*$)/) ||
        die2 "didn't find end pragma at column $col in " .
             "\"$str\", line $line of $file";
    $newRight = $1 . "BECAUSE " . $warning . " " . $2;

    $modifiedLine = $newLeft . $newRight;
    $contents[$line-1] = $modifiedLine . "\n";
    $changedOutput = 1;
    print STDOUT "$inputline\n";
}

if ($hasSource) {
    $newSourcefilename = $sourcefilename.".annotationRemover.tmp";
    open(NEWSOURCE, "> $newSourcefilename") || \
	die2 "can't open $newSourcefilename";
    print NEWSOURCE @contents;
    close NEWSOURCE;
    rename $newSourcefilename,$sourcefilename || \
	die2 "can't update $sourcefilename";
}

exit $changedOutput;
