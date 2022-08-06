#check library and install packages
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(fields)) install.packages("fields", repos = "http://cran.us.r-project.org")
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("knitr", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) install.packages("kableExtra", repos = "http://cran.us.r-project.org")
if(!require(grid)) install.packages("grid", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(lattice)) install.packages("lattice", repos = "http://cran.us.r-project.org")
if(!require(gridExtra)) install.packages("gridExtra", repos = "http://cran.us.r-project.org")
if(!require(recosystem)) install.packages("recosystem", repos = "http://cran.us.r-project.org")
options(timeout=100)
# Load library
library(caret)
library(data.table)
library(fields)
library(tidyverse)
library(knitr)
library(kableExtra)
library(grid)
library(ggplot2)
library(lattice)
library(gridExtra)
library(recosystem)

##Data Clean

#create tempfile and download
dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

#read the file and give name to columns
ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

#create data frame movielens
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

#remove temporary files
rm(dl, ratings, movies)

##Data exploration and visualization 

# "The first 5 rows of the dataset, movielens"
knitr::kable(head(movielens |> as_tibble(),5),
             caption = "The first 5 rows of the dataset movielens",
             digits = 2,
             align = "cccccc",
             position = "b") |>
  kable_styling(latex_options = "scale_down")


#matrix users x movies x rating
users <- sample(unique(movielens$userId), 50)
movielens %>% filter(userId %in% users) %>% 
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>% select(sample(ncol(.), 50)) %>% 
  as.matrix() %>% t(.) %>%
  image.plot(1:50, 1:50,. , 
             xlab="Movies", 
             ylab="Users", 
             main= "Matrix: users x movies x rating")

# Unique users, movies, rating
p0 <- tableGrob(movielens %>% summarize(n_users = n_distinct(userId), 
                                  n_movies = n_distinct(movieId),
                                  n_rating = length(rating)))

# Top 5 movies
p1 <- movielens %>%
  group_by(title) %>%
  summarize(count = n()) %>%
  arrange(-count) %>%
  top_n(5, count) %>%
  ggplot(aes(count, reorder(title, count))) +
  geom_bar(color = "black", fill = "#999999", stat = "identity") +
  geom_text(aes(label=count), position=position_dodge(width=0.9), hjust=1.5) +
  xlab("Number of Ratings") +
  ylab("Movies") +
  theme_bw()

# Top 5 users
p2 <- movielens %>%
  group_by(userId) %>%
  summarize(count = n()) %>%
  arrange(-count) %>%
  top_n(5, count) %>%
  ggplot(aes(count, reorder(userId, count))) +
  geom_bar(color = "black", fill = "#999999", stat = "identity") +
  geom_text(aes(label=count), position=position_dodge(width=0.9), hjust=1.5) +
  xlab("Number of Ratings") +
  ylab("Users") +
  theme_bw()

#Top 5 most rating movies and users, unique variables
gridExtra::grid.arrange(p1,
               arrangeGrob (p0, p2, ncol = 2), 
               nrow = 2, 
               top = "Top 5 most rating movies and users, unique variables")

#distribution of movies and user
p3 <- movielens %>% 
  count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 100, color = "black", fill="#999999") + 
  scale_x_log10() + 
  ggtitle("Movies")
p4 <- movielens %>% 
  count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 100, color = "black", fill="#999999") + 
  scale_x_log10() + 
  ggtitle("Users")
gridExtra::grid.arrange(p3, p4, nrow = 2)

##Modeling approach

# RMSE
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#create data partition train_set and temp
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
train_set <- movielens[-test_index,]
temp <- movielens[test_index,]

# Matching userId and movieId in both train and test sets
test_set <- temp %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")

# Adding back rows into train set
removed <- anti_join(temp, test_set)
train_set <- rbind(train_set, removed)

#remove temporary file
rm(test_index, temp, removed)

##naive approach
#Average of all ratings
mu <- mean(train_set$rating)
mu

# Model 1 - Naive_rmse
naive_rmse <- RMSE(test_set$rating, mu)
results <- tibble(Method = "Model 1: Naive RMSE", RMSE = naive_rmse)
knitr::kable( results %>% as_tibble(), caption = "RMSE by Approach") %>% 
  kable_styling(font_size = 8)

##Bias approach

# Movie bias
movies_bias <- train_set %>%
  group_by(movieId) %>%
  summarize(b_i = mean(rating - mu))

# Movie bias distribution
movies_bias %>% ggplot(aes(b_i)) +
  geom_histogram(color = "black", fill = "#999999", bins = 100) +
  xlab("Movie Bias") +
  ylab("Count") +
  theme_bw()

# Model 2 - Movie Bias RMSE
predicted_ratings <- mu + test_set %>%
  left_join(movies_bias, by = "movieId") %>%
  pull(b_i)
m_bias_rmse <- RMSE(predicted_ratings, test_set$rating)
results <- bind_rows(results, tibble(Method = "Model 2: Mean + movie bias",
                                     RMSE = m_bias_rmse))
knitr::kable( results %>% as_tibble(), caption = "RMSE by Approach") %>% 
  kable_styling(font_size = 8)


# User bias
user_bias <- train_set %>%
  left_join(movies_bias, by = "movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

# Model 3 - Movies and User Bias RMSE
predicted_ratings <- test_set %>%
  left_join(movies_bias, by = "movieId") %>%
  left_join(user_bias, by = "userId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  pull(pred)
movie_user_bias_rmse <- RMSE(predicted_ratings, test_set$rating)
results <- bind_rows(results, 
                     tibble(Method = "Model 3: Mean + movie bias + user effect",
                            RMSE = movie_user_bias_rmse))
knitr::kable( results %>% as_tibble(), caption = "RMSE by Approach") %>% 
  kable_styling(font_size = 8)

## Regularization approach

# Model 4 Regularization and RMSE
lambda <- 20
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))
movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda), n_i = n()) 

#Regularized x original
tibble(original = movie_avgs$b_i, 
       regularlized = movie_reg_avgs$b_i, 
       n = movie_reg_avgs$n_i) %>%
  ggplot(aes(original, regularlized, size=sqrt(n))) + 
  geom_point(shape=1, alpha=0.5)

#minimize the full model
lambdas <- seq(0, 10, 0.25)

rmses <- sapply(lambdas, function(x){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+x))
  b_u <- train_set %>%
    left_join(b_i, by = "movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+x))
  predicted_ratings <- test_set %>%
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

# Lambdas plot and best parameter
qplot(lambdas, rmses, color = I("#000000"))
lambda <- lambdas[which.min(rmses)]
lambda

# Model 4 - Regularization RMSE
results <- bind_rows(results, 
                     tibble(Method = "Model 4: Regularized movie and user effects",
                            RMSE = min(rmses)))
knitr::kable( results %>% as_tibble(), caption = "RMSE by Approach") %>% 
  kable_styling(font_size = 8)


## Matrix Factorization approach

# Model 5 Matrix Factorization using recosystem
set.seed(1, sample.kind="Rounding")
train_reco <- with(train_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
test_reco <- with(test_set, data_memory(user_index = userId, item_index = movieId, rating = rating))
r <- Reco()

# Recosystem tuning
opts <- r$tune(train_reco, opts = list(dim = c(20, 30),
                                            costp_l2 = c(0.01, 0.1),
                                            costq_l2 = c(0.01, 0.1),
                                            lrate = c(0.01, 0.1),
                                            nthread = 1,
                                            niter = 10))



# Recosystem training
r$train(train_reco, opts = c(opts$min, nthread = 1, niter = 30))

# Recosystem prediction
results_reco <- r$predict(test_reco, out_memory())

# Model 5 RMSE
factorization_rmse <- RMSE(results_reco, test_set$rating)
results <- bind_rows(results, tibble(Method = "Model 5: Matrix factorization using recosystem", RMSE = factorization_rmse))
knitr::kable( results %>% as_tibble(), caption = "RMSE by Approach") %>% 
  kable_styling(font_size = 20)

##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

dl <- tempfile()
download.file("https://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)
rm(dl, ratings, movies, test_index, temp, movielens, removed)

################################################
#End of inital code supplied by HarvardX course
################################################


# Final validation
set.seed(1, sample.kind="Rounding")
edx_reco <- with(edx, data_memory(user_index = userId, item_index = movieId, rating = rating))
validation_reco <- with(validation, data_memory(user_index = userId, item_index = movieId, rating = rating))
r <- Reco()

para_reco <- r$tune(edx_reco, opts = list(dim = c(20, 30),
                                          costp_l2 = c(0.01, 0.1),
                                          costq_l2 = c(0.01, 0.1),
                                          lrate = c(0.01, 0.1),
                                          nthread = 4,
                                          niter = 10))

r$train(edx_reco, opts = c(para_reco$min, nthread = 4, niter = 30))

# Final prediction
final_reco <- r$predict(validation_reco, out_memory())

# Final RMSE
final_rmse <- RMSE(final_reco, validation$rating)
results <- bind_rows(results, tibble(Method = "Final validation: Matrix factorization using recosystem", RMSE = final_rmse))
results %>% knitr::kable()

## Appendix
# For future debugging purposes
print("Version Info")
version

update.packages(ask = FALSE, checkBuilt = TRUE)
tinytex::tlmgr_update()

