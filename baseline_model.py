from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error
import pandas as pd
from typing import List
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import ElasticNet


class baseline_model():
    def __init__(self, data, x_cols_numeric: List[str], x_cols_categorical: List[str], y_target: str, seed: int
                 ):
        self.data = data
        self.x_cols_numeric = x_cols_numeric
        self.x_cols_categorical = x_cols_categorical
        self.y_target = y_target
        self.seed = seed
        # Initiate items
        self.x_final = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.Y_train = pd.DataFrame()
        self.Y_test = pd.DataFrame()
        self.baseline_results = pd.DataFrame()
        self.feature_imp_df = pd.DataFrame()
        # baseline models
        self.regression_model = None
        self.forest_model = None
        self.xgb_model = None
        self.decision_model = None
        self.elastic_model = None

    def encode_categorical(self):
        # Encode categorical features using OneHotEncoder - this is the default encoding method
        encoder = OneHotEncoder(sparse_output=False)
        encoded_data = encoder.fit_transform(self.data[self.x_cols_categorical])
        # Create a DataFrame with the encoded data
        encoded_columns = encoder.get_feature_names_out(self.x_cols_categorical)
        encoded_df = pd.DataFrame(encoded_data, columns=encoded_columns)

        # Combine categorical with numerical
        self.x_final = pd.concat([self.data[self.x_cols_numeric], encoded_df], axis=1)
        print("-- Total number of rows in X:", self.x_final.shape[0])
        print("-- Total number of columns in X:", self.x_final.shape[1])

    def train_test_split(self, train_split_ratio=0.8):
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.x_final,
                                                                                self.data[self.y_target],
                                                                                test_size=1 - train_split_ratio,
                                                                                random_state=self.seed)
        print("-- Number of rows in train:", self.X_train.shape[0])
        print("-- Number of rows in test:", self.X_test.shape[0])

    def regression_fit(self):
        model = LinearRegression()
        model.fit(self.X_train, self.Y_train)
        self.regression_model = model

    def decision_fit(self):
        model = DecisionTreeRegressor(random_state=self.seed)
        model.fit(self.X_train, self.Y_train)
        self.decision_model = model

    def random_forest_fit(self):
        model = RandomForestRegressor(random_state=self.seed)
        model.fit(self.X_train, self.Y_train)
        self.forest_model = model

    def xgb_fit(self):
        model = XGBRegressor(random_state=self.seed)
        model.fit(self.X_train, self.Y_train)
        self.xgb_model = model

    def elastic_fit(self):
        model = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=self.seed)
        model.fit(self.X_train, self.Y_train)
        self.elastic_model = model

    def model_predict(self, model):
        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.X_test)

        return y_pred_train, y_pred_test

    def evaluate_train(self, y_pred_train):
        mse = mean_squared_error(self.Y_train, y_pred_train)
        r2 = r2_score(self.Y_train, y_pred_train)
        rmse = root_mean_squared_error(self.Y_train, y_pred_train)
        return mse, r2, rmse

    def evaluate_test(self, y_pred_test):
        mse = mean_squared_error(self.Y_test, y_pred_test)
        r2 = r2_score(self.Y_test, y_pred_test)
        rmse = root_mean_squared_error(self.Y_test, y_pred_test)
        return mse, r2, rmse

    def get_feature_imp(self, model):
        best_feature_importance = pd.DataFrame(model.feature_importances_)
        feature_names = pd.DataFrame(self.x_final.columns)
        feature_importance_mapped = pd.concat([feature_names, best_feature_importance], axis=1)
        feature_importance_mapped.columns = ['Feature', 'Feature_Importance']
        feature_importance_mapped["Model"] = str(model)
        return feature_importance_mapped

    def fit_all(self):
        self.encode_categorical()
        self.train_test_split()
        self.regression_fit()
        self.elastic_fit()
        self.decision_fit()
        self.random_forest_fit()
        self.xgb_fit()

        regression_y_pred_train, regression_y_pred_test = self.model_predict(self.regression_model)
        elastic_y_pred_train, elastic_y_pred_test = self.model_predict(self.elastic_model)
        decision_y_pred_train, decision_y_pred_test = self.model_predict(self.decision_model)
        forest_y_pred_train, forest_y_pred_test = self.model_predict(self.forest_model)
        xgb_y_pred_train, xgb_y_pred_test = self.model_predict(self.xgb_model)

        eval_list = ["regression", "elastic", "decision", "forest", "xgb"]

        for item in eval_list:
            mse_train, r2_train, rmse_train = self.evaluate_train(eval(item + "_y_pred_train"))
            mse_test, r2_test, rmse_test = self.evaluate_test(eval(item + "_y_pred_test"))

            result_i = pd.DataFrame([{"model": item,
                                      "MSE Train": mse_train,
                                      "MSE Test": mse_test,
                                      "RMSE Train": rmse_train,
                                      "RMSE Test": rmse_test,
                                      "R^2 Train": r2_train,
                                      "R^2 Test": r2_test,
                                      }])

            self.baseline_results = pd.concat([self.baseline_results, result_i], axis=0)
            if item not in ["regression","elastic"]:
                model_feature_imp = self.get_feature_imp(eval("self."+item+"_model"))
                self.feature_imp_df = pd.concat([self.feature_imp_df, model_feature_imp], axis=0)
