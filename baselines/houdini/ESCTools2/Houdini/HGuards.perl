#!/usr/bin/perl

my $fileno = -1;

if ($ARGV[0] =~ /^(\d+)/) {
  $fileno = $1;
} else {
  print STDERR "Command-line argument didn't start with fileno\n";
  exit(-1);
}

while (<STDIN>) {
  print if (/^G_$fileno\./)
}
