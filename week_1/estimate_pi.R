rm(list=ls())

pi.est <- function(n)
{
    est <- 0
    for(i in 1:n)
    {
        temp <- 4 / (i*2 - 1)
        if(i %% 2 == 0)
        {
            est <- est - temp
        } else est <- est + temp
    }
    return(est)
}

n <- 1000
estimates <- numeric(n)

for(i in 1:n)
{
    estimates[i] <- pi.est(i)
}
par(font.lab=2, font.axis=2, cex=1.5)
show.xy <- seq(100, n, by=19)
plot(show.xy, estimates[show.xy], type='l', xlab='Repetition no.',
     ylab='Estimated value', main='Estimating pi')
