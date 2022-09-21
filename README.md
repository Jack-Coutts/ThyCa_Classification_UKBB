# Classification of thyriod cancer patients in the UK Biobank database using supervised machine learning methods

<p align="center">
<img src="https://github.com/Jack-Coutts/ThyCa_Classification_UKBB/blob/main/visualisations/thyca_class_conf.png" width=75% height=75% class="center">
</p>

***Abstract:*** *Thyroid cancer is the most common endocrine malignancy worldwide and has reached epidemic levels in recent years. Thyroid cancer is notoriously difficult to diagnose as it typically becomes detectable as thyroid nodules with only 5-15% of nodules being cancerous. The ubiquity of benign thyroid nodules means that further diagnostic techniques like high-resolution ultrasonography and fine needle aspiration biopsies must be used to diagnose the disease. These techniques have limitations in that they are labour intensive, invasive, dependent on the skill of the clinician, and have accuracies ranging from 70-97%. The widely described problems of thyroid cancer overdiagnosis and overtreatment have frequently been attributed to the limitations of current diagnostic techniques and it is clear there is a need for improved thyroid cancer diagnostic techniques.*

*The use of machine learning techniques in biomedical research is on the rise and diagnosis by machine learning models is becoming available for an increasing number of conditions. This work looked to utilise three machine learning models using phenotype data from UK Biobank to address the limitations of current diagnostic techniques and classify individuals with thyroid cancer. The random forest, support vector machine, and multilayer perceptron models evaluated were all found to perform poorly on the dataset. These results suggest that for thyroid cancer diagnosis, the phenotype features in the UK Biobank do not contain enough information about the thyroid for the predictive models to succeed.*

## Data Source 

This research was conducted using data from UK Biobank, a major biomedical database: [www.ukbiobank.ac.uk](https://www.ukbiobank.ac.uk/).

Specifically, this work used binary disease data for 394,884 individuals and 787 disease as well as phenotype data for 390 phenotypes and 454,119 individuals. The phenotypes used included lifestyle factors, biological samples e.g.blood samples, socioeconomic factors, family history, physical measures e.g.height, and genetic principle components.

*Reference:*

Sudlow, C. et al. (2015) ‘UK Biobank: An Open Access Resource for Identifying the Causes of a Wide Range of Complex Diseases of Middle and Old Age’, PLoS Medicine, 12(3), p. e1001779. Available at: https://doi.org/10.1371/journal.pmed.1001779.
