using Plots
step = plot(-7.:.01:7.,[repeat([0],700);repeat([1],701)],title="x>0",lw=3)
step_inv = plot(-7.:.01:7.,[repeat([0],700);repeat([1],701)][end:-1:1],title="x<0",lw=3)
f(x) = exp(x)/(exp(x)+1)
sigm = plot(-7.:.1:7.,f.(Vector(-7.:.1:7.)),title="Sigmoid(x)",lw=3)
sigm_inv = plot(-7.:.1:7.,f.(Vector(-7.:.1:7.))[end:-1:1],title="Sigmoid(-x)",lw=3)
comp = plot(step,step_inv,sigm,sigm_inv,layout=(2,2),legend=false)
png(comp, "comp.png")
