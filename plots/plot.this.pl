#!/usr/bin/env perl -w

use strict;

if (scalar @ARGV != 1) { die "Usage: $0 <num>\n"; }

my $n = $ARGV[0];

my $outfile = "plot.valid" . $n . ".R";

print STDERR "Writing to $outfile\n";

open(OUT,">",$outfile) or die "Cannot open $outfile for writing.\n";
print OUT "x <- read.table(\"valid" . $n . ".txt\");";
print OUT "png(file=\"valid$n.png\",width=800,height=800);";
print OUT "plot(x)\n";
print OUT "dev.off()\n";

close(OUT);

