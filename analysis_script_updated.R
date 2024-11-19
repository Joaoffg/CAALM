<<<<<<< HEAD
# Load necessary libraries
library(dplyr)
library(tidyr)
library(afex)

# Read the CSV file
data <- read.csv("all_results.csv")

# Create a CAALM indicator variable
data$CAALM <- ifelse(data$Context == "with_context", TRUE, FALSE)

# Create a unique identifier for each run
data$GroupID <- with(data, paste(Model, Dataset, Sample.size, Seed, sep = "_"))

# Ensure that for each GroupID, we have both CAALM and non-CAALM runs
paired_counts <- data %>%
  group_by(GroupID) %>%
  summarise(n = n()) %>%
  filter(n == 2)

# Filter data to include only paired runs
data_paired <- data %>% filter(GroupID %in% paired_counts$GroupID)

# Split data into CAALM and non-CAALM
data_caalm <- data_paired %>% filter(CAALM == TRUE)
data_noncaalm <- data_paired %>% filter(CAALM == FALSE)

# Merge the two datasets on GroupID
data_merged <- merge(
  data_caalm,
  data_noncaalm,
  by = "GroupID",
  suffixes = c("_caalm", "_noncaalm")
)

# Calculate mean values for the metrics
mean_f1_macro_caalm <- mean(data_merged$eval_f1_macro_caalm)
mean_f1_macro_noncaalm <- mean(data_merged$eval_f1_macro_noncaalm)

mean_accuracy_balanced_caalm <- mean(data_merged$eval_accuracy_balanced_caalm)
mean_accuracy_balanced_noncaalm <- mean(data_merged$eval_accuracy_balanced_noncaalm)

mean_accuracy_standard_caalm <- mean(data_merged$eval_accuracy_not_b_caalm)
mean_accuracy_standard_noncaalm <- mean(data_merged$eval_accuracy_not_b_noncaalm)

# Perform paired t-tests
t_test_f1_macro <- t.test(
  data_merged$eval_f1_macro_caalm,
  data_merged$eval_f1_macro_noncaalm,
  paired = TRUE
)

t_test_accuracy_balanced <- t.test(
  data_merged$eval_accuracy_balanced_caalm,
  data_merged$eval_accuracy_balanced_noncaalm,
  paired = TRUE
)

t_test_accuracy_standard <- t.test(
  data_merged$eval_accuracy_not_b_caalm,
  data_merged$eval_accuracy_not_b_noncaalm,
  paired = TRUE
)

# Create a SmallSample indicator (sample size <= 2500)
data_merged$SmallSample <- data_merged$Sample.size_caalm <= 2500

# Calculate differences in metrics
data_merged <- data_merged %>%
  mutate(
    diff_f1_macro = eval_f1_macro_caalm - eval_f1_macro_noncaalm,
    diff_accuracy_balanced = eval_accuracy_balanced_caalm - eval_accuracy_balanced_noncaalm,
    diff_accuracy_standard = eval_accuracy_not_b_caalm - eval_accuracy_not_b_noncaalm
  )

# Calculate mean differences overall and for small samples
mean_diff_f1_macro <- mean(data_merged$diff_f1_macro)
mean_diff_accuracy_balanced <- mean(data_merged$diff_accuracy_balanced)
mean_diff_accuracy_standard <- mean(data_merged$diff_accuracy_standard)

mean_diff_f1_macro_small <- mean(data_merged$diff_f1_macro[data_merged$SmallSample == TRUE])
mean_diff_accuracy_balanced_small <- mean(data_merged$diff_accuracy_balanced[data_merged$SmallSample == TRUE])
mean_diff_accuracy_standard_small <- mean(data_merged$diff_accuracy_standard[data_merged$SmallSample == TRUE])

# Reshape data to long format for ANOVA
data_long <- data_merged %>%
  select(
    GroupID,
    Model_caalm,
    Dataset_caalm,
    Sample.size_caalm,
    Seed_caalm,
    SmallSample,
    eval_f1_macro_caalm,
    eval_f1_macro_noncaalm,
    eval_accuracy_balanced_caalm,
    eval_accuracy_balanced_noncaalm,
    eval_accuracy_not_b_caalm,
    eval_accuracy_not_b_noncaalm
  ) %>%
  gather(
    key = "Metric",
    value = "Value",
    eval_f1_macro_caalm,
    eval_f1_macro_noncaalm,
    eval_accuracy_balanced_caalm,
    eval_accuracy_balanced_noncaalm,
    eval_accuracy_not_b_caalm,
    eval_accuracy_not_b_noncaalm
  ) %>%
  mutate(
    CAALM = ifelse(grepl("_caalm$", Metric), TRUE, FALSE),
    Metric = sub("_caalm$", "", Metric),
    Metric = sub("_noncaalm$", "", Metric),
    GroupID = as.factor(GroupID),
    CAALM = as.factor(CAALM),
    SmallSample = as.factor(SmallSample)
  )

# Repeated measures ANOVA for SmallSample interaction
anova_f1_macro <- aov_ez(
  id = "GroupID",
  dv = "Value",
  data = filter(data_long, Metric == "eval_f1_macro"),
  within = "CAALM",
  between = "SmallSample"
)

anova_accuracy_balanced <- aov_ez(
  id = "GroupID",
  dv = "Value",
  data = filter(data_long, Metric == "eval_accuracy_balanced"),
  within = "CAALM",
  between = "SmallSample"
)

anova_accuracy_standard <- aov_ez(
  id = "GroupID",
  dv = "Value",
  data = filter(data_long, Metric == "eval_accuracy_not_b"),
  within = "CAALM",
  between = "SmallSample"
)

# Create a MilitaryDataset indicator
data_long$MilitaryDataset <- data_long$Dataset_caalm == "Military"
data_long$MilitaryDataset <- as.factor(data_long$MilitaryDataset)

# Repeated measures ANOVA for MilitaryDataset interaction
anova_f1_macro_military <- aov_ez(
  id = "GroupID",
  dv = "Value",
  data = filter(data_long, Metric == "eval_f1_macro"),
  within = "CAALM",
  between = "MilitaryDataset"
)

anova_accuracy_balanced_military <- aov_ez(
  id = "GroupID",
  dv = "Value",
  data = filter(data_long, Metric == "eval_accuracy_balanced"),
  within = "CAALM",
  between = "MilitaryDataset"
)

anova_accuracy_standard_military <- aov_ez(
  id = "GroupID",
  dv = "Value",
  data = filter(data_long, Metric == "eval_accuracy_not_b"),
  within = "CAALM",
  between = "MilitaryDataset"
)

# Output results
# Paired t-tests results
print("Paired t-test for F1 Macro:")
print(t_test_f1_macro)

print("Paired t-test for Balanced Accuracy:")
print(t_test_accuracy_balanced)

print("Paired t-test for Standard Accuracy:")
print(t_test_accuracy_standard)

# Mean differences
print(paste("Overall mean difference in F1 Macro:", mean_diff_f1_macro))
print(paste("Mean difference in F1 Macro for small samples:", mean_diff_f1_macro_small))

print(paste("Overall mean difference in Balanced Accuracy:", mean_diff_accuracy_balanced))
print(paste("Mean difference in Balanced Accuracy for small samples:", mean_diff_accuracy_balanced_small))

print(paste("Overall mean difference in Standard Accuracy:", mean_diff_accuracy_standard))
print(paste("Mean difference in Standard Accuracy for small samples:", mean_diff_accuracy_standard_small))

# ANOVA results
print("Repeated measures ANOVA for F1 Macro with SmallSample interaction:")
print(anova_f1_macro)

print("Repeated measures ANOVA for Balanced Accuracy with SmallSample interaction:")
print(anova_accuracy_balanced)

print("Repeated measures ANOVA for Standard Accuracy with SmallSample interaction:")
print(anova_accuracy_standard)

print("Repeated measures ANOVA for F1 Macro with MilitaryDataset interaction:")
print(anova_f1_macro_military)

print("Repeated measures ANOVA for Balanced Accuracy with MilitaryDataset interaction:")
print(anova_accuracy_balanced_military)

print("Repeated measures ANOVA for Standard Accuracy with MilitaryDataset interaction:")
print(anova_accuracy_standard_military)
=======
# Load necessary libraries
library(dplyr)
library(tidyr)
library(afex)

# Read the CSV file
data <- read.csv("all_results.csv")

# Create a CAALM indicator variable
data$CAALM <- ifelse(data$Context == "with_context", TRUE, FALSE)

# Create a unique identifier for each run
data$GroupID <- with(data, paste(Model, Dataset, Sample.size, Seed, sep = "_"))

# Ensure that for each GroupID, we have both CAALM and non-CAALM runs
paired_counts <- data %>%
  group_by(GroupID) %>%
  summarise(n = n()) %>%
  filter(n == 2)

# Filter data to include only paired runs
data_paired <- data %>% filter(GroupID %in% paired_counts$GroupID)

# Split data into CAALM and non-CAALM
data_caalm <- data_paired %>% filter(CAALM == TRUE)
data_noncaalm <- data_paired %>% filter(CAALM == FALSE)

# Merge the two datasets on GroupID
data_merged <- merge(
  data_caalm,
  data_noncaalm,
  by = "GroupID",
  suffixes = c("_caalm", "_noncaalm")
)

# Calculate mean values for the metrics
mean_f1_macro_caalm <- mean(data_merged$eval_f1_macro_caalm)
mean_f1_macro_noncaalm <- mean(data_merged$eval_f1_macro_noncaalm)

mean_accuracy_balanced_caalm <- mean(data_merged$eval_accuracy_balanced_caalm)
mean_accuracy_balanced_noncaalm <- mean(data_merged$eval_accuracy_balanced_noncaalm)

mean_accuracy_standard_caalm <- mean(data_merged$eval_accuracy_not_b_caalm)
mean_accuracy_standard_noncaalm <- mean(data_merged$eval_accuracy_not_b_noncaalm)

# Perform paired t-tests
t_test_f1_macro <- t.test(
  data_merged$eval_f1_macro_caalm,
  data_merged$eval_f1_macro_noncaalm,
  paired = TRUE
)

t_test_accuracy_balanced <- t.test(
  data_merged$eval_accuracy_balanced_caalm,
  data_merged$eval_accuracy_balanced_noncaalm,
  paired = TRUE
)

t_test_accuracy_standard <- t.test(
  data_merged$eval_accuracy_not_b_caalm,
  data_merged$eval_accuracy_not_b_noncaalm,
  paired = TRUE
)

# Create a SmallSample indicator (sample size <= 2500)
data_merged$SmallSample <- data_merged$Sample.size_caalm <= 2500

# Calculate differences in metrics
data_merged <- data_merged %>%
  mutate(
    diff_f1_macro = eval_f1_macro_caalm - eval_f1_macro_noncaalm,
    diff_accuracy_balanced = eval_accuracy_balanced_caalm - eval_accuracy_balanced_noncaalm,
    diff_accuracy_standard = eval_accuracy_not_b_caalm - eval_accuracy_not_b_noncaalm
  )

# Calculate mean differences overall and for small samples
mean_diff_f1_macro <- mean(data_merged$diff_f1_macro)
mean_diff_accuracy_balanced <- mean(data_merged$diff_accuracy_balanced)
mean_diff_accuracy_standard <- mean(data_merged$diff_accuracy_standard)

mean_diff_f1_macro_small <- mean(data_merged$diff_f1_macro[data_merged$SmallSample == TRUE])
mean_diff_accuracy_balanced_small <- mean(data_merged$diff_accuracy_balanced[data_merged$SmallSample == TRUE])
mean_diff_accuracy_standard_small <- mean(data_merged$diff_accuracy_standard[data_merged$SmallSample == TRUE])

# Reshape data to long format for ANOVA
data_long <- data_merged %>%
  select(
    GroupID,
    Model_caalm,
    Dataset_caalm,
    Sample.size_caalm,
    Seed_caalm,
    SmallSample,
    eval_f1_macro_caalm,
    eval_f1_macro_noncaalm,
    eval_accuracy_balanced_caalm,
    eval_accuracy_balanced_noncaalm,
    eval_accuracy_not_b_caalm,
    eval_accuracy_not_b_noncaalm
  ) %>%
  gather(
    key = "Metric",
    value = "Value",
    eval_f1_macro_caalm,
    eval_f1_macro_noncaalm,
    eval_accuracy_balanced_caalm,
    eval_accuracy_balanced_noncaalm,
    eval_accuracy_not_b_caalm,
    eval_accuracy_not_b_noncaalm
  ) %>%
  mutate(
    CAALM = ifelse(grepl("_caalm$", Metric), TRUE, FALSE),
    Metric = sub("_caalm$", "", Metric),
    Metric = sub("_noncaalm$", "", Metric),
    GroupID = as.factor(GroupID),
    CAALM = as.factor(CAALM),
    SmallSample = as.factor(SmallSample)
  )

# Repeated measures ANOVA for SmallSample interaction
anova_f1_macro <- aov_ez(
  id = "GroupID",
  dv = "Value",
  data = filter(data_long, Metric == "eval_f1_macro"),
  within = "CAALM",
  between = "SmallSample"
)

anova_accuracy_balanced <- aov_ez(
  id = "GroupID",
  dv = "Value",
  data = filter(data_long, Metric == "eval_accuracy_balanced"),
  within = "CAALM",
  between = "SmallSample"
)

anova_accuracy_standard <- aov_ez(
  id = "GroupID",
  dv = "Value",
  data = filter(data_long, Metric == "eval_accuracy_not_b"),
  within = "CAALM",
  between = "SmallSample"
)

# Create a MilitaryDataset indicator
data_long$MilitaryDataset <- data_long$Dataset_caalm == "Military"
data_long$MilitaryDataset <- as.factor(data_long$MilitaryDataset)

# Repeated measures ANOVA for MilitaryDataset interaction
anova_f1_macro_military <- aov_ez(
  id = "GroupID",
  dv = "Value",
  data = filter(data_long, Metric == "eval_f1_macro"),
  within = "CAALM",
  between = "MilitaryDataset"
)

anova_accuracy_balanced_military <- aov_ez(
  id = "GroupID",
  dv = "Value",
  data = filter(data_long, Metric == "eval_accuracy_balanced"),
  within = "CAALM",
  between = "MilitaryDataset"
)

anova_accuracy_standard_military <- aov_ez(
  id = "GroupID",
  dv = "Value",
  data = filter(data_long, Metric == "eval_accuracy_not_b"),
  within = "CAALM",
  between = "MilitaryDataset"
)

# Output results
# Paired t-tests results
print("Paired t-test for F1 Macro:")
print(t_test_f1_macro)

print("Paired t-test for Balanced Accuracy:")
print(t_test_accuracy_balanced)

print("Paired t-test for Standard Accuracy:")
print(t_test_accuracy_standard)

# Mean differences
print(paste("Overall mean difference in F1 Macro:", mean_diff_f1_macro))
print(paste("Mean difference in F1 Macro for small samples:", mean_diff_f1_macro_small))

print(paste("Overall mean difference in Balanced Accuracy:", mean_diff_accuracy_balanced))
print(paste("Mean difference in Balanced Accuracy for small samples:", mean_diff_accuracy_balanced_small))

print(paste("Overall mean difference in Standard Accuracy:", mean_diff_accuracy_standard))
print(paste("Mean difference in Standard Accuracy for small samples:", mean_diff_accuracy_standard_small))

# ANOVA results
print("Repeated measures ANOVA for F1 Macro with SmallSample interaction:")
print(anova_f1_macro)

print("Repeated measures ANOVA for Balanced Accuracy with SmallSample interaction:")
print(anova_accuracy_balanced)

print("Repeated measures ANOVA for Standard Accuracy with SmallSample interaction:")
print(anova_accuracy_standard)

print("Repeated measures ANOVA for F1 Macro with MilitaryDataset interaction:")
print(anova_f1_macro_military)

print("Repeated measures ANOVA for Balanced Accuracy with MilitaryDataset interaction:")
print(anova_accuracy_balanced_military)

print("Repeated measures ANOVA for Standard Accuracy with MilitaryDataset interaction:")
print(anova_accuracy_standard_military)
>>>>>>> 53c9b4e0ed2869a8c28af646c60a06b6f826806d
