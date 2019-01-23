"""
This is an attempt to train simple regression tree.
Idea is taken from the University Washington course of ML foundation hosted on Coursera.
"""
import numpy as np
import pandas as pd


def compute_sse(actual, predicted):
    ans = np.sum(np.square(actual-predicted))
    return ans


def direct_rmse_error(actual, predicted):
    ans = np.sqrt(np.sum(np.square(actual-predicted))/len(actual))
    return ans


def get_r2_score(actual, predicted):
    ybar = np.sum(actual)/len(actual)
    ssreg = np.sum((predicted-ybar)**2)
    sstot = np.sum((actual - ybar)**2)
    r2 = ssreg / sstot
    return r2


class DecisionTree:

    def __init__(self, min_sample_size = 1000, max_depth=10):
        self.min_sample_size = min_sample_size
        self.max_depth = max_depth
        self.tree = None


    def create_leaf(self, data, target):
        ans = np.mean(data[target])
        leaf = {'splitting_feature': None,
                'left': None,
                'right': None,
                'prediction':ans,
                'is_leaf':True}
        return leaf


    @staticmethod
    def find_best_splitting_feature(data, features, target):

        ans = np.inf

        for f in features:

            unique_values = data[f].unique()

            if len(unique_values) <=10 :
                # optimization for features than have discrete values
                all_values = set(data[f])
                all_values = list(all_values)
            else:
                all_values = data[f].sort_values().values

            if len(all_values)==1:
                continue

            thresholds = set()
            for i in range(len(all_values)-1):
                mid_point = np.mean([all_values[i], all_values[i+1]])
                thresholds.add(mid_point)

            for mm in thresholds:

                left_data = data[data[f]<=mm]
                right_data = data[data[f]>mm]

                left_pred = np.mean(data[target])
                right_pred = np.mean(data[target])

                left_sse = compute_sse(left_data[target], left_pred)
                right_sse = compute_sse(right_data[target], right_pred)

                total_sse = left_sse + right_sse

                if total_sse < ans:
                    ans = total_sse
                    splitting_feature = f
                    split_value = mm
                    to_remove = True if len(data[f].dropna().unique())<=2 else False

        return (splitting_feature, split_value, to_remove)


    def decision_tree_create(self, data, features, target, current_depth = 0):
        # We can keep splitting on continuous predictor multiple times

        print ("Current depth", current_depth, "Data points", data.shape[0])

        if data.shape[0]<=self.min_sample_size:
            print ("\t\tReached min sample size")
            return self.create_leaf(data, target)

        if current_depth==self.max_depth:
            print ("\t\tReached max depth")
            return self.create_leaf(data, target)

        if len(features)==0:
            print ("\t\t All features used")
            return self.create_leaf(data, target)

        splitting_feature, split_value, to_remove = self.find_best_splitting_feature(data, features, target)

        if not splitting_feature:
            # Can happend when all features have uniform values
            return self.create_leaf(data, target)

        print ("Splitting on", splitting_feature, "with splitting value =", split_value)
        left_data = data[data[splitting_feature]<=split_value]
        right_data = data[data[splitting_feature]>split_value]

        if to_remove:
            features.remove(splitting_feature)

        if len(left_data)==0 or len(right_data)==0:
            print ("\t\tEither left or right child has 0 length")
            return self.decision_tree_create(data, features, target, current_depth)


        if len(left_data) == len(data):
            left_tree = self.create_leaf(left_data, target)
        else:
            left_tree = self.decision_tree_create(left_data, features, target, current_depth + 1)

        if len(right_data)==len(data):
            right_tree = self.create_leaf(right_data, target)
        else:
            right_tree = self.decision_tree_create(right_data, features, target, current_depth + 1)

        return {'is_leaf': False,
                'prediction': None,
                'splitting_feature': splitting_feature,
                'split_value':split_value,
                'left': left_tree,
                'right': right_tree}


    def classify(self, tree, row):
        if tree['is_leaf']:
            return tree['prediction']
        else:
            splitting_feature = tree['splitting_feature']
            row_val = row[splitting_feature]
            split_value = tree['split_value']
            if row_val<=split_value:
                return self.classify(tree['left'], row)
            else:
                return self.classify(tree['right'], row)

    def fit(self, data, features, target):
        self.tree = self.decision_tree_create(data, features, target, current_depth=0)

    def predict(self, data):
        y_pred = []
        for i, row in data.iterrows():
            ans = self.classify(self.tree, row)
            y_pred.append(ans)
        return y_pred




def get_data():
    df = pd.read_csv('trainingData.csv')
    df = df[df.loan_amount<20000]
    del df['Id']
    df2 = pd.get_dummies(df)
    df2 = df2.dropna()
    return df2



def main():

    df2 = get_data()

    df2 = df2.sample(frac=1, random_state=29)
    split_length = np.floor_divide(len(df2) * 8, 10)
    train_df = df2[:split_length]
    test_df = df2[split_length:]

    target = 'loan_amount'

    features = [x for x in df2.columns if x!=target]
    features = np.random.choice(features, 10, replace=False)  # Keeping just 10 feratures to speed up computation
    features = set(features)
    features.add('age')
    features.add('house_area')
    features.add('young_dependents')
    features.add('water_availabity')
    features.add('annual_income')
    features.add('social_class_G.C')
    features.add('secondary_business_Daily wage labourer')
    features.add('loan_installments')
    # features = set(features)

    clf = DecisionTree()
    clf.fit(train_df, features, target)
    y_pred = clf.predict(test_df)

    np_y_test = test_df[target].values
    rmse = direct_rmse_error(np_y_test, y_pred)
    base_rmse = direct_rmse_error(np_y_test, np.mean(np_y_test))
    r2 = get_r2_score(np_y_test, y_pred)

    print(rmse, base_rmse, r2)



if __name__=="__main__":
    main()
