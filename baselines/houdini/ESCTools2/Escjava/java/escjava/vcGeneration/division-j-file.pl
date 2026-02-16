#!/usr/bin/perl -w
#=============================================================================
#
# writeTo
#
# $Revision$
# $Date$
# $Source$
#
# Copyright (c) 2005, Patrice Chalin (chalin@cse.concordia.ca)
#
#=============================================================================

$^W = 1; # equivalent to using the -w command-line flag under UNIX.

use strict;
use diagnostics;
use Getopt::Std;
use FileHandle;
use Data::Dumper;

#------------------------------------------------------------------------------
# Global var decl

my $gProgNm    = "writeTo";            # program name
my $gDbgTr     = 0;
my $gVerbose   = 0;
my $gVersion   = "1.0a1";
my $gProgRslt  = 0;

my $gPath;

#------------------------------------------------------------------------------
# Forward declarations

sub cmdLine();
sub main();
sub HelpAndExit(;$);
sub Usage();

sub getFileHandle($$);

#------------------------------------------------------------------------------
# Prog body

main();
exit $gProgRslt;

#------------------------------------------------------------------------------
# Subroutines
#------------------------------------------------------------------------------

sub main()
{
   autoflush STDOUT, 1;
   autoflush STDERR, 1;
   my $args = cmdLine();
   processInput();
}

#------------------------------------------------------------------------------
# Application specific utils

sub processInput() {
    my $header = "";
    my $fh;
    my $n = 0;
    while(<>) {
	printf STDERR "." if $gDbgTr;
	my ($className) = /class (\w+)/;
	if(defined $className) {
	    my $fileName = "$className.java";
	    if($fh) {
		printf ", %d lines\n", $n if $gVerbose;
		$fh->close();
		$fh = 0;
	    }
	    if($fileName) {
		$fileName = "$gPath/$fileName" if $gPath;
		$fh = getFileHandle($fileName, ">");
		printf "%s", $fileName if $gVerbose;
		$n = 0;
		print $fh $header;
	    }
	} else {
	    $n++;
	}
	if($fh) {
	    print $fh $_;
	} else {
	    $header .= $_;
	}
    }
}

#------------------------------------------------------------------------------
# General utils

sub getFileHandle($$)
{
   my ($path, $mode) = @_;

   my $fh = new FileHandle "$mode $path";
   if(!$fh) {
   printf("getFileHandle: could not open file '$path'.\n");
   return;
   }
   return $fh;
}

sub cmdLine()
{
   # Process command line flags
   my %opt;

   if(!getopts('f:hp:vZ:',\%opt)) {
   HelpAndExit();
   } elsif($opt{h}) {
   Usage();
   exit 0;
   }

   $gVerbose = defined $opt{v};
   $gDbgTr ||= $opt{Z};
   $gPath = $opt{p} if $opt{p};

   # Take up remaining arguments as file names
   if(!$opt{f}) {
   my @files = @ARGV;
   $opt{files} = \@files;
   } else {
   # Ensure that there are no other arguments lingering around.
   HelpAndExit('extra arguments on the command line: "'.
   join(' ', @ARGV) . '".') if @ARGV;
   }
   return \%opt;
}

sub HelpAndExit(;$)
{
   my ($msg) = @_;
   printf STDERR "$gProgNm: %s\n", $msg if $msg;
   printf STDERR "$gProgNm: use -h flag for help.\n";
   exit ($gProgRslt || 1);
}

sub Usage()
{
   print <<EndOfUsageText;
Usage: $gProgNm [options] [files]
e.g.
  $gProgNm -v

 The purpose of this script is to write blocks of lines from the input
to the
 named files. Blocks are delimited with lines containing the text
writeTo="<fn>"

 Options:
  -p <path>  prefix file names with the given path.
  -h         print this help information.
  -v         verbose (quiet by default).
  -Z <n>     set debugging trace to level <n>.

[Script version $gVersion]
EndOfUsageText
}

#------------------------------------------------------------------------------
# End of file $RCSfile$
