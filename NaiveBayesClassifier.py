import numpy as np
import pandas as pd
import math
import sys

class NaiveBayesBasketballClassifier:
    def __init__(self):
        self.classes = None
        self.prior_probabilities = {}
        self.likelihood = {}
        self.statistics = {}
        self.target = 'label'  # whether the home team won (1) or lost (0)
        self.categorical_features = {'home_wl_pre5', 'away_wl_pre5'}  # the two categories
        self.numerical_features = {
            'min_avg5', 'fg_pct_home_avg5', 'fg3_pct_home_avg5', 'ft_pct_home_avg5',
            'creb_home_avg5', 'dreb_home_avg5', 'reb_home_avg5', 'ast_home_avg5', 
            'stl_home_avg5', 'blk_home_avg5', 'tov_home_avg5', 'pf_home_avg5',
            'pts_home_avg5', 'fg_pct_away_avg5', 'fg3_pct_away_avg5', 'ft_pct_away_avg5',
            'creb_away_avg5', 'dreb_away_avg5', 'reb_away_avg5', 'ast_away_avg5', 
            'stl_away_avg5', 'blk_away_avg5', 'tov_away_avg5', 'pf_away_avg5', 'pts_away_avg5'
        }

    def fit(self, data):
        self.classes = np.unique(data[self.target])
        
        # Calculate priors and likelihoods
        for c in self.classes:
            class_data = data[data[self.target] == c]
            self.prior_probabilities[c] = len(class_data) / len(data)
            self.likelihood[c] = {}
            self.statistics[c] = {}

            for feature in data.columns:
                if feature == self.target:
                    continue

                if feature in self.categorical_features:
                    # Calculate likelihood for categorical features
                    values = class_data[feature].value_counts(normalize=True).to_dict()
                    self.likelihood[c][feature] = values
                elif feature in self.numerical_features:
                    # Calculate mean and std for numerical features
                    mean = class_data[feature].mean()
                    std = class_data[feature].std()
                    self.statistics[c][feature] = (mean, std)

    def normalpdf(self, x, mean, std):
        if std == 0: std = 1e-10 
        return (1 / (math.sqrt(2 * math.pi) * std)) * math.exp(-((x - mean) ** 2 / (2 * std ** 2)))

    def predict(self, data):
        predictions = []
        for _, row in data.iterrows():
            class_probs = {}
            for c in self.classes:
                # Start with prior probability
                current_probability = self.prior_probabilities[c]

                for feature in data.columns:
                    if feature == self.target:
                        continue

                    value = row[feature]
                    if feature in self.categorical_features:
                        # Use categorical likelihood
                        feature_probs = self.likelihood[c].get(feature, {})
                        current_probability *= feature_probs.get(value, 1 / len(data[feature].unique()))
                    elif feature in self.numerical_features:
                        # Use normal PDF 
                        mean, std = self.statistics[c][feature]
                        current_probability *= self.normalpdf(value, mean, std)

                class_probs[c] = current_probability

            predicted_class = max(class_probs, key=class_probs.get)
            predictions.append(predicted_class)
        
        return predictions

def main(training_path, validation_path):
    training_data = pd.read_csv(training_path)
    
    model = NaiveBayesBasketballClassifier()
    model.fit(training_data)

    validation_data = pd.read_csv(validation_path)
    predictions = model.predict(validation_data)

    for p in predictions:
        print(p)

if __name__ == "__main__":
    training_path = sys.argv[1]
    validation_path = sys.argv[2]
    main(training_path, validation_path)
