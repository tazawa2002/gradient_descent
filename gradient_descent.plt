set terminal gif animate optimize delay 1 size 800,600
set output 'gradient_descent.gif'

stats 'gradient_descent.dat' nooutput

set key opaque
set key box
set key spacing 1.5

# set xrange [-2:2]
# set yrange [-1:3]
set cbrange [0:2500]
# set palette defined (0 "#440154", 0.2 "#482878", 0.4 "#3E4A89", 0.6 "#2A788E", 0.8 "#22A884", 1 "#FDE725")
set palette defined (0 "#FDE725", 0.2 "#22A884", 0.4 "#2A788E", 0.6 "#3E4A89", 0.8 "#482878", 1 "#440154")
set logscale cb
set title "Gradient Descent"
set xlabel "x"
set ylabel "y"
set cblabel "Cost"
set size ratio -1
set grid

# ヒートマップのプロット
set pm3d map

step = 10

do for [i=0:(STATS_records-1)/step] {
    splot 'heatmap.dat' using 1:2:3 notitle with pm3d, \
          'gradient_descent.dat' index 0 using 2:3:(0) every ::0::((i*step>0)?(i*step-1):0) with points pt 7 ps 0.5 lc rgb "white" notitle, \
          'gradient_descent.dat' index 0 using 2:3:(0) every ::i*step::i*step with points pt 7 ps 1 lc rgb "black" title sprintf("Step %d", i*step)
}
