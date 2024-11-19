<<<<<<< HEAD
# Load necessary libraries
library(dplyr)
library(tidyr)

# Read the CSV file
data <- read.csv("nemo_results.csv")

# Filter data to include only "with_context" and "with_nemo" conditions
data_filtered <- data %>%
  filter(Context %in% c("with_context", "with_nemo"))

# Create a unique identifier for each run (excluding Context)
data_filtered$RunID <- with(data_filtered, paste(Dataset, Sample.size, Seed, sep = "_"))

# Check that we have both "with_context" and "with_nemo" for each RunID
paired_runs <- data_filtered %>%
  group_by(RunID) %>%
  summarise(n = n()) %>%
  filter(n == 2)

# Filter data to include only paired runs
data_paired <- data_filtered %>% filter(RunID %in% paired_runs$RunID)

# Reshape data to wide format
data_wide <- data_paired %>%
  select(RunID, Context, eval_f1_macro, eval_accuracy_balanced, eval_accuracy_not_b) %>%
  pivot_wider(names_from = Context, values_from = c(eval_f1_macro, eval_accuracy_balanced, eval_accuracy_not_b))

# Perform paired t-tests
t_test_f1_macro <- t.test(
  data_wide$eval_f1_macro_with_context,
  data_wide$eval_f1_macro_with_nemo,
  paired = TRUE
)

t_test_accuracy_balanced <- t.test(
  data_wide$eval_accuracy_balanced_with_context,
  data_wide$eval_accuracy_balanced_with_nemo,
  paired = TRUE
)

t_test_accuracy_standard <- t.test(
  data_wide$eval_accuracy_not_b_with_context,
  data_wide$eval_accuracy_not_b_with_nemo,
  paired = TRUE
)

# Calculate mean differences
mean_diff_f1_macro <- mean(data_wide$eval_f1_macro_with_context - data_wide$eval_f1_macro_with_nemo)
mean_diff_accuracy_balanced <- mean(data_wide$eval_accuracy_balanced_with_context - data_wide$eval_accuracy_balanced_with_nemo)
mean_diff_accuracy_standard <- mean(data_wide$eval_accuracy_not_b_with_context - data_wide$eval_accuracy_not_b_with_nemo)

# Output results
print("Paired t-test for F1 Macro (with_context vs. with_nemo):")
print(t_test_f1_macro)

print("Paired t-test for Balanced Accuracy (with_context vs. with_nemo):")
print(t_test_accuracy_balanced)

print("Paired t-test for Standard Accuracy (with_context vs. with_nemo):")
print(t_test_accuracy_standard)

print(paste("Mean difference in F1 Macro:", mean_diff_f1_macro))
print(paste("Mean difference in Balanced Accuracy:", mean_diff_accuracy_balanced))
print(paste("Mean difference in Standard Accuracy:", mean_diff_accuracy_standard))
=======
# Load necessary libraries
library(dplyr)
library(tidyr)

# Read the CSV file
data <- read.csv("nemo_results.csv")

# Filter data to include only "with_context" and "with_nemo" conditions
data_filtered <- data %>%
  filter(Context %in% c("with_context", "with_nemo"))

# Create a unique identifier for each run (excluding Context)
data_filtered$RunID <- with(data_filtered, paste(Dataset, Sample.size, Seed, sep = "_"))

# Check that we have both "with_context" and "with_nemo" for each RunID
paired_runs <- data_filtered %>%
  group_by(RunID) %>%
  summarise(n = n()) %>%
  filter(n == 2)

# Filter data to include only paired runs
data_paired <- data_filtered %>% filter(RunID %in% paired_runs$RunID)

# Reshape data to wide format
data_wide <- data_paired %>%
  select(RunID, Context, eval_f1_macro, eval_accuracy_balanced, eval_accuracy_not_b) %>%
  pivot_wider(names_from = Context, values_from = c(eval_f1_macro, eval_accuracy_balanced, eval_accuracy_not_b))

# Perform paired t-tests
t_test_f1_macro <- t.test(
  data_wide$eval_f1_macro_with_context,
  data_wide$eval_f1_macro_with_nemo,
  paired = TRUE
)

t_test_accuracy_balanced <- t.test(
  data_wide$eval_accuracy_balanced_with_context,
  data_wide$eval_accuracy_balanced_with_nemo,
  paired = TRUE
)

t_test_accuracy_standard <- t.test(
  data_wide$eval_accuracy_not_b_with_context,
  data_wide$eval_accuracy_not_b_with_nemo,
  paired = TRUE
)

# Calculate mean differences
mean_diff_f1_macro <- mean(data_wide$eval_f1_macro_with_context - data_wide$eval_f1_macro_with_nemo)
mean_diff_accuracy_balanced <- mean(data_wide$eval_accuracy_balanced_with_context - data_wide$eval_accuracy_balanced_with_nemo)
mean_diff_accuracy_standard <- mean(data_wide$eval_accuracy_not_b_with_context - data_wide$eval_accuracy_not_b_with_nemo)

# Output results
print("Paired t-test for F1 Macro (with_context vs. with_nemo):")
print(t_test_f1_macro)

print("Paired t-test for Balanced Accuracy (with_context vs. with_nemo):")
print(t_test_accuracy_balanced)

print("Paired t-test for Standard Accuracy (with_context vs. with_nemo):")
print(t_test_accuracy_standard)

print(paste("Mean difference in F1 Macro:", mean_diff_f1_macro))
print(paste("Mean difference in Balanced Accuracy:", mean_diff_accuracy_balanced))
print(paste("Mean difference in Standard Accuracy:", mean_diff_accuracy_standard))
>>>>>>> 53c9b4e0ed2869a8c28af646c60a06b6f826806d
