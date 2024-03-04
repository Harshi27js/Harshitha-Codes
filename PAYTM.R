
library(tidyverse)
library(ggthemes)
library(forecast)
library(tseries)
library(gridExtra)
library(rugarch)
setwd("C:\\Users\\harsh\\OneDrive\\Desktop\\SCMA")
getwd()
df = read.csv("PAYTM.csv")

names(df)
head(df)
tail(df)
df$Date <- as.Date(df$Date, format = '%m-%d-%Y')
df= df[order(df$Date),]
head(df)
View(df)
df$Price <- as.numeric(df$Price)
df$Open <- as.numeric(df$Open)
df$High <- as.numeric(df$High)
df$Low <- as.numeric(df$Low)
missing_values <- is.na(df$Price)
missing_values_open <- is.na(df$Open)
missing_values_high <- is.na(df$High)
missing_values_low <- is.na(df$Low)
df$Price[missing_values] <- mean(df$Price, na.rm = TRUE)
df$Open[missing_values_open] <- mean(df$Open, na.rm = TRUE)
df$High[missing_values_high] <- mean(df$High, na.rm = TRUE)
df$Low[missing_values_low] <- mean(df$Low, na.rm = TRUE)

plot(df$Price)
plot(df$Open)
plot(df$High)
plot(df$Low)
model.arima = auto.arima(df$Price , max.order = c(3 , 1 ,3) , stationary = TRUE , trace = T , ic = 'aicc')
model.arima
model.arima$residuals %>% ggtsdisplay(plot.type = 'hist' , lag.max = 14)
ar.res = model.arima$residuals
Box.test(model.arima$residuals , lag = 14 , fitdf = 2 , type = 'Ljung-Box')

tsdisplay(ar.res^2 , main = 'Squared Residuals')

model.spec = ugarchspec(variance.model = list(model = 'sGARCH', garchOrder = c(1, 1)),
                        mean.model = list(armaOrder = c(0, 0)))
model.fit = ugarchfit(spec = model.spec, data = ar.res, solver = 'solnp')

options(scipen = 999)
model.fit@fit$matcoef

jarque.bera.test(ar.res)
quantile(ar.res, 0.05)
qplot(ar.res, geom = 'histogram') + geom_histogram(fill = 'lightblue' , bins = 30) +
  geom_histogram(aes(ar.res[ar.res < quantile(ar.res , 0.05)]) , fill = 'red' , bins = 30) +
  labs(x = 'Daily Returns')


p2_1 = qplot(ar.res , geom = 'density') + geom_density(fill = 'blue' , alpha = 0.4) + 
  geom_density(aes(rnorm(200000 , 0 , sd(ar.res))) , fill = 'red' , alpha = 0.25) + 
  labs(x = 'Daily Returns')

p2_2 = qplot(p2_1 = qplot(ar.res , geom = 'density') + geom_density(fill = 'blue' , alpha = 0.4) + 
               geom_density(aes(rnorm(200000 , 0 , sd(ar.res))) , fill = 'red' , alpha = 0.25) + 
               labs(x = ''), geom = 'density') + geom_density(fill = 'blue' , alpha = 0.4) + 
  geom_density(aes(rnorm(200000 , 0 , sd(ar.res))) , fill = 'red' , alpha = 0.25) + 
  coord_cartesian(xlim = c(-0.07 , -0.02) , ylim = c(0 , 10)) + 
  geom_vline(xintercept = c(qnorm(p = c(0.01 , 0.05) , mean = mean(ar.res) , sd = sd(ar.res))) , 
             color = c('darkgreen' , 'green') , size = 1) + labs(x = 'Daily Returns')

grid.arrange(p2_1 , p2_2 , ncol = 1)

fitdist(distribution = 'std' , x = ar.res)$pars
cat("For a = 0.05 the quantile value of normal distribution is: " , 
    qnorm(p = 0.05) , "\n" ,
    "For a = 0.05 the quantile value of t-distribution is: " ,
    qdist(distribution = 'std' , shape = 2.0100001 , p = 0.05) , "\n" , "\n" , 
    'For a = 0.01 the quantile value of normal distribution is: ' , 
    qnorm(p = 0.01) , "\n" , 
    "For a = 0.01 the quantile value of t-distribution is: " , 
    qdist(distribution = 'std' , shape = 2.0100001 , p = 0.01) , sep = "")

qplot(x = 1:407, y = ar.res, geom = 'point') +
  geom_point(colour = 'lightgrey', size = 2) +
  geom_line(aes(x = 1:407, y = model.fit@fit$sigma * (-1.644854)), colour = 'red') +
  geom_hline(yintercept = sd(ar.res) * qnorm(0.05), colour = 'darkgreen', size = 1.2) +
  theme_light() +
  labs(x = '', y = 'Daily Returns', title = 'Value at Risk Comparison')
