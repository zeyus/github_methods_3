data(mtcars)
plot(X[, 2], Y, xlab = 'Weight (1000 lbs)', ylab='Miles per gallon',
     main='Scatter plot (mtcars)', ylim=c(0, 40))

model <- lm(mpg ~ wt + 1, data=mtcars)
## The general linear model
# Y = X x beta + epsilon
Y <- model$model$mpg ## observations
X <- model.matrix(model) ## design matrix
epsilon <- model$residuals ## residuals

# beta unknown, estimate beta.hat
X.T <- t(X) ## transpose
beta.hat <- solve(X.T %*% X) %*% X.T %*% Y ## "solve" is inverse, 
                                           ## "%*%" is matrix multiplication
# compare beta.hat with model$coefficients
print(model)
print(beta.hat)

par(font.lab=2, font.axis=2, cex=1.2)
plot(X[, 2], Y, xlab = 'Weight (1000 lbs)', ylab='Miles per gallon',
     main='Linear regression (mtcars)', ylim=c(0, 40))
abline(model) ## plots the regressor line

## add residual lines
n.obs <- length(Y)
for(index in 1:n.obs)
{
    x1 <- X[index, 2]
    x2 <- x1
    y1 <- beta.hat[1] +  beta.hat[2] * x1
    y2 <- Y[index]
    lines(c(x1, x2), c(y1, y2), lty=3)
}

## exercise: build the appropriate X for an ANOVA of with "am" as a factorial effect