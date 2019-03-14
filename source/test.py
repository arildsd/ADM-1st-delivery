s = r"""Name & Comment \\ [0.5ex] \hline\hline
 is_gender_female & 1 if the student is female, 0 otherwise \\ \hline
 is_gender_male & 1 if the student is male, 0 otherwise \\ \hline
 is_race_group A & 1 if the student belongs to race A, 0 otherwise \\ \hline
 is_race_group B & 1 if the student belongs to race B, 0 otherwise \\ \hline
 is_race_group C & 1 if the student belongs to race C, 0 otherwise \\ \hline
 is_race_group D & 1 if the student belongs to race D, 0 otherwise \\ \hline
 is_race_group E & 1 if the student belongs to race E, 0 otherwise \\ \hline
 is_parent_education_associate's degree & 1 if the student's parent(s) have an associate's degree, 0 otherwise \\ \hline
 is_parent_education_education_bachelor's degree & 1 if the student's parent(s) have a education_bachelor's degree, 0 otherwise \\ \hline
 is_parent_education_high school & 1 if the student's parent(s) have a high school degree, 0 otherwise \\ \hline
 is_parent_education_master's degree & 1 if the student's parent(s) have a master's degree, 0 otherwise \\ \hline
 is_parent_education_some college & 1 if the student's parent(s) have a some college experience, 0 otherwise \\ \hline
 is_parent_education_some high school & 1 if the student's parent(s) have a some high school experience, 0 otherwise \\ \hline
 is_lunch_free/reduced & 1 if the student has free or reduced lunch, 0 otherwise \\ \hline
 is_lunch_standard & 1 if the student has standard lunch, 0 otherwise \\ \hline
 is_prepared_completed & 1 if the student attended a test preparation course, 0 otherwise \\ \hline
 is_prepared_none & 1 if the student didn't attended a test preparation course, 0 otherwise \\ \hline"""

s = s.replace("_", "\\_")
print(s)