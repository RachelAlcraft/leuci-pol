

# Create a vector of values 1-20 inc
data <- c(1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) 

# calc mean and sd
# convert distribution to z distribution
z_scores <- (data-mean(data))/sd(data)

# output details
z_scores


"#############################################"
# Some dummy data
data <- c(0,0,0,0,0,1,1,12) 
# calc mean and sd and convert distribution to z distribution
z_scores <- (data-mean(data))/sd(data)
# output details
z_scores
mean(data)
sd(data)
# The values at zero are now at a negative value - the data should be transposed so zero is mainatined
zero = (0-mean(data))/sd(data)
z_scores <- (z_scores-zero)
# output details
z_scores
mean(data)
sd(data)

