library(ggplot2)
data <- read.csv("Mali-B.csv")
head(data)
x <- data$Malignant	       
y <- data$B.cell

data <- data.frame(x=x, y=y)

cor_result <- cor.test(data$x, data$y)
R_val <- cor_result$estimate
P_val <- cor_result$p.value
P_val_sci <- formatC(P_val, format = "e", digits = 3)

x_pos <- max(data$x) * 0.5
y_pos <- max(data$y) * 0.9


ggplot(data, aes(x=x, y=y)) +
    geom_point(color="steelblue", alpha=0.8) +
    geom_smooth(method="lm", color="steelblue", fill="steelblue", alpha=0.2) +
    
    annotate("text", x = x_pos, y = y_pos, 
             label = paste0("R = ", round(R_val, 4), ", P = ", P_val_sci),
             color="steelblue", size=6, face = "bold") +
    
    theme_classic() +
    scale_y_continuous(limits = c(min(data$y) , max(data$y))) +
    scale_x_continuous(limits = c(min(data$x) , max(data$x))) +
    
    theme(
        axis.title = element_text(size = 12, face = "bold"),  
        axis.text = element_text(size = 12),                 
        panel.border = element_rect(color = "black", fill = NA, size = 1.5)  
    ) +
    labs(x = "Proportion of B Cells", y = "Proportion of GC B Cells")

get_clean_stats <- function(x, y) {
  res <- cor.test(x, y)
  cat(sprintf("t = %.2f, r = %.2f [%.2f, %.2f], P = %s\n", 
              res$statistic, 
              res$estimate, 
              res$conf.int[1], 
              res$conf.int[2], 
              formatC(res$p.value, format = "e", digits = 2)))
}

get_clean_stats(x,y)

