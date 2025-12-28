
- The case filter should also look at variable names.


Visualization for the plots:
- [ ] Right now, I have an annotation for each of the cases that gives detail, but perhaps it would be best to just let the axis label spill over into the plot itself (avoiding any line wrapping). I'll need to move the axis labels up a bit to make sure they don't fully cover the plots.
 - Think I can do this by setting the y-position of the plots and then also adding custom y-ticks. However, I want to make it so that the tick labels spill over onto the plot.
- [ ] I need to put the failed runs separately. Red 'x' on the rightmost part of the plot.
- [ ] xmax should be benchmark-specific! Right now it uses the values for all the benchmarks
