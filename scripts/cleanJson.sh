# to run on the output of bits solver, to get the stats as a json array
awk '/stats_start/{f = 1} {if (f) toPr = toPr $0 "\n"} /stats_end/ {print toPr; f = 0}' | sed '$d' | sed '/stats_start/d;s/^c//;s/stats_end/,/;$s/.*/]/;2s/^/[/'
