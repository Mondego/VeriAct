#!/usr/bin/perl

# Copyright 2000, 2001, Compaq Computer Corporation

# Copyright (c) 1999, Compaq Computer Corporation
# Change history:
#   31 Aug 1999  rustan & flanagan  Created

$lastWarning = "no warning";

while (<STDIN>) {
  chop;
  if ($_ =~ /Warning:/) {
      $lastWarning = $_;
  } elsif ($_ =~ /^Associated declaration is "(.*)":(.*)\.(.*):/) {
      print STDOUT "$1 $2 $3 $lastWarning\n";
  }
}
close(STDIN);
