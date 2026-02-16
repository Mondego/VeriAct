#!/usr/local/bin/perl
# Copyright (c) 1999, Compaq Computer Corporation
# Change history:
#   31 Aug 1999  rustan & flanagan  Created

$lastWarning = "no warning";

while (<STDIN>) {
  chop;
  if ($_ =~ /Warning:/) {
      $lastWarning = $_;
  } elsif ($_ =~ /^Associated declaration is "(.*)", line (.*), col (.*):/) {
      print STDOUT "$1 $2 $3 $lastWarning\n";
  }
}
close(STDIN);
