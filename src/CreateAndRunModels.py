import yaml
import sys
import time
import datetime
import yaml
import pandas as pd
import numpy as np
sys.path.append('../')
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.cross_decomposition import PLSRegression
from skopt import BayesSearchCV
from src.DataCleansing import Cleansing_CreateModel
from src.Models import model_param_grid_selection
from src.OverSampling import SMOTE_gen

def LoadModelWithParams(instance_name):
    loaded_instance = pickle.load(open(f'../preserved_instances/{instance_name}','rb'))
    return loaded_instance

class CreateAndRunModel():
    def __init__(self, X_df, y_df, file_name = 'NoFileName'):
        # 設定の読み込み
        with open('../config/config.yaml', 'r', encoding = 'utf-8') as file:
            self.config = yaml.safe_load(file)
        with open('../config/config_create_model.yaml', 'r', encoding = 'utf-8') as file:
            self.config_create = yaml.safe_load(file)
        # ファイル名(拡張子まで)
        self.file_name = file_name
        # 目的変数
        self.y_df = pd.DataFrame(y_df)
        self.scaler_y = self.config_create['learning']['scaler_y']
        # 説明変数            
        self.X_df = pd.DataFrame(X_df)
        self.cleansing_method = self.config_create['feature_engineering']['cleansing']['method']
        self.X_df_new, self.del_variable_numbers_li = self.X_df, []
        self.X_df_new, self.del_variable_numbers_li = Cleansing_CreateModel(self.X_df, self.y_df)
        self.scaler_X = self.config_create['learning']['scaler_X']
        self.X_std = -10000
        self.X_mean = -10000
        self.y_std = -10000
        self.y_mean = -10000
        # 学習条件
        self.loss_function = self.config_create['learning']['loss_function']
        self.method_of_cross_valid = self.config_create['learning']['cross_validation']
        self.random_state = self.config_create['learning']['random_state']
        self.param_search_method = self.config_create['learning']['param_search']
        self.learning_option = self.config_create['learning']['option']
        self.PLS_components = -1
        self.PCA_components = -1
        # 結果を入れるリスト
        self.X_test_li = []
        self.y_pred_li = []
        self.y_test_li = []
        self.best_params_li = []
        self.reports_li = [] # レポート作成用文字列
        # 作成したモデル
        self.created_model = None
        # 日付
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        self.date_str = (f'{now:%Y%m%d}')
        # リストへ書き込み
        self.reports_li.append(self.date_str)
        self.reports_li.append('')
        self.reports_li.append(f'CSV File: {self.file_name}')
        self.reports_li.append('')
        self.reports_li.append('======= X_df ============================================================')
        self.reports_li.append(self.X_df)
        self.reports_li.append('')
        self.reports_li.append(self.X_df.describe())
        self.reports_li.append('')
        self.reports_li.append('======= y ===============================================================')
        self.reports_li.append(self.y_df)
        self.reports_li.append('')
        self.reports_li.append(self.y_df.describe())
        self.reports_li.append('')
        self.reports_li.append(f'======= X_Preprocessing =====================================================')
        self.reports_li.append(f'Cleansing Method: {self.cleansing_method}')
        
        # ここから特徴量選択
        if self.cleansing_method == 'Val':
            threshold_of_Val = self.config_create['feature_engineering']['cleansing']['same_value_threshold']
            self.reports_li.append(f'                  └── Threshold of Val: {threshold_of_Val}')
        elif self.cleansing_method == 'Val_Corr':
            threshold_of_Val = self.config_create['feature_engineering']['cleansing']['same_value_threshold']
            threshold_of_Corr = self.config_create['feature_engineering']['cleansing']['correlation_coefficient_threshold']
            self.reports_li.append(f'                  └── Threshold of Val: {threshold_of_Val}, Threshold of Corr: {threshold_of_Corr}')
        elif self.cleansing_method == 'Chi_Square':
            threshold_of_p_value = self.config_create['feature_engineering']['cleansing']['p_value_of_chi_square']
            self.reports_li.append(f'                  └── Threshold of p value : {threshold_of_p_value}')
        elif self.cleansing_method == 'ANOVA':
            threshold_of_p_value = self.config_create['feature_engineering']['cleansing']['p_value_of_ANOVA']
            self.reports_li.append(f'                  └── Threshold of p value : {threshold_of_p_value}')
        elif self.cleansing_method == 'Stepwise_OLS':
            threshold_of_p_value_in = self.config_create['feature_engineering']['cleansing']['p_value_of_OLS_in']
            threshold_of_p_value_out = self.config_create['feature_engineering']['cleansing']['p_value_of_OLS_out']
            self.reports_li.append(f'                  └── Threshold of p value : {threshold_of_p_value_in} (in), {threshold_of_p_value_out} (out)')
        # ここまで
        
        self.reports_li.append('')
        self.reports_li.append(f'Deleted Variable Numbers: {self.del_variable_numbers_li}')
        self.reports_li.append('')
        self.reports_li.append('======= preprocessed X_df ====================================================')
        self.reports_li.append(self.X_df_new)
        self.reports_li.append('')
       
    def ScaleAndTrain(self, selected_model):
        self.selected_model = selected_model
        self.model, self.params = model_param_grid_selection(selected_model)
        self.reports_li.append(f'======= Learning Conditions =================================================')
        self.reports_li.append(f'Model: {self.selected_model}')
        self.reports_li.append(f'Hyper Parameters: {self.params}')
        self.reports_li.append(f'Loss Function: {self.loss_function}')
        self.reports_li.append(f'X_Scaler: {self.scaler_X}')
        self.reports_li.append(f'y_Scaler: {self.scaler_y}')
        self.reports_li.append('')
        self.reports_li.append(f'=============================================================================')
        self.reports_li.append('')
        score_dict = {'MAE': 'neg_mean_absolute_error', 'MSE': 'neg_mean_squared_error', 'RMSE': 'neg_root_mean_squared_error', 'RMSLE': 'neg_root_mean_squared_log_error', 'R2': 'r2'}
        scoring = score_dict[self.loss_function]
        self.X_mean = self.X_df_new.mean()
        if self.scaler_X == 'S_Standard':
            self.X_std = self.X_df_new.std(ddof=0)
            scaled_X = (self.X_df_new - self.X_mean) / self.X_std
        elif self.scaler_X == 'U_Standard':
            self.X_std = self.X_df_new.std(ddof=1)
            scaled_X = (self.X_df_new - self.X_mean) / self.X_std
        else:
            print('X 標準化なし')
            scaled_X = self.X_df_new
        if self.method_of_cross_valid == 'k-Fold':
            n_splits = self.config_create['learning']['k-Fold'] 
            cv_splitter = KFold(n_splits = n_splits, shuffle=True, random_state = self.random_state)
        elif self.method_of_cross_valid == 'LOO':
            cv_splitter = KFold(n_splits=len(self.X_df_new), shuffle=True, random_state = self.random_state)
        else:
            raise ValueError(f"Error in inner CV")
        t1 = time.time()
        y = self.y_df.values.reshape(-1)
        # SMOTE # 元データは除かない!!
        if self.learning_option == 'SMOTE':
            X_resampled_df, y_resampled_df = SMOTE_gen(pd.concat([self.X_df_new, self.y_df], axis = 1), self.y_df.columns.values[0])
            self.SMOTE_df = pd.concat([X_resampled_df, y_resampled_df], axis = 1)
            self.reports_li.append(f'======= SMOTE resampled_X ===================================================')
            self.reports_li.append(X_resampled_df)
            self.reports_li.append('')
            self.reports_li.append(X_resampled_df.describe())
            self.reports_li.append('')
            self.reports_li.append(f'======= SMOTE resampled_y ===================================================')
            self.reports_li.append(y_resampled_df)
            self.reports_li.append('')
            self.reports_li.append(y_resampled_df.describe())
            self.reports_li.append('')
            a, b = self.config_create['learning']['SMOTE'], self.config_create['learning']['SMOTE_threshold']
            self.reports_li.append(f'Method: {a} SMOTE, Threshold: {b}')
            self.reports_li.append('')
            if self.scaler_X == 'S_Standard' or self.scaler_X == 'U_Standard':
                scaled_X_resampled = (X_resampled_df - self.X_mean) / self.X_std
            else:
                scaled_X_resampled = X_resampled_df
            # scaled_Xとyに代入
            scaled_X = scaled_X_resampled
            y = y_resampled_df.values.reshape(-1)
        self.y_mean = y.mean()    
        if self.scaler_y == 'S_Standard':
            print('!!標本標準偏差を用いています（不偏標準偏差を使うべきではないですか？）')
            self.y_std = y.std(ddof=0)
            scaled_y = (y - self.y_mean) / self.y_std
        elif self.scaler_y == 'U_Standard':
            Y_std_tmp = y.std(ddof=1)
            scaled_y = (y - self.y_mean) / self.y_std
        else:
            print('Y 標準化なし')
            scaled_y = y
        if self.param_search_method == 'Grid':
            param_search_cv = GridSearchCV(estimator = self.model, param_grid = self.params, cv = cv_splitter, scoring = scoring)
        elif self.param_search_method == 'Bayes':
            n_iter = self.config['learning']['Bayes_n_iter']
            param_search_cv = BayesSearchCV(estimator = self.model, search_spaces = self.params, cv = cv_splitter, scoring = scoring, n_iter = n_iter)
        else:
            raise ValueError(f"Error in param_search")
        if self.learning_option == 'PLS':
            self.pls = PLSRegression(n_components = self.config_create['learning']['PLS_components'])
            self.pls.fit(scaled_X, scaled_y)
            pls_X = self.pls.transform(scaled_X)
            param_search_cv.fit(pls_X, scaled_y)
            df_param_search_cv_tmp = pd.DataFrame(param_search_cv.cv_results_)
            df_param_search_cv = df_param_search_cv_tmp.sort_values('rank_test_score', ascending = True)
            self.reports_li.append(df_param_search_cv.loc[:,['rank_test_score','params','mean_test_score']].head(3))
            self.reports_li.append('')
            print(param_search_cv.best_params_)
            self.reports_li.append(param_search_cv.best_params_)
            self.reports_li.append('')
            self.created_model = param_search_cv.best_estimator_
            self.created_model.fit(pls_X, scaled_y)
            print('model has been created!!')
            print('created model: self.created_model')
            print('')
            self.reports_li.append('model has been created!!')
            self.reports_li.append('created model: self.created_model')
            self.reports_li.append('')
        else:
            param_search_cv.fit(scaled_X, scaled_y)
            df_param_search_cv_tmp = pd.DataFrame(param_search_cv.cv_results_)
            df_param_search_cv = df_param_search_cv_tmp.sort_values('rank_test_score', ascending = True)
            self.reports_li.append(df_param_search_cv.loc[:,['rank_test_score','params','mean_test_score']].head(3))
            self.reports_li.append('')
            print(param_search_cv.best_params_)
            self.reports_li.append(param_search_cv.best_params_)
            self.created_model = param_search_cv.best_estimator_
            self.created_model.fit(scaled_X, scaled_y)
            print('model has been created!!')
            print('created model: self.created_model')
            print('')
            self.reports_li.append('model has been created!!')
            self.reports_li.append('created model: self.created_model')
            self.reports_li.append('')
        tn = time.time()
        print('Cumulative Time: {:.2f} seconds'.format(tn - t1))
        print()
        self.reports_li.append('Cumulative Time: {:.2f} seconds'.format(tn - t1))
        self.reports_li.append('')
        self.reports_li.append('======= End of Learning =====================================================')
        self.reports_li.append('')
        return
    
    def Predict(self, X_test_df, key_word='NoKeyWord'):
        self.X_test_li = []
        self.y_pred_li = []
        t1 = time.time()
        X_test = X_test_df.loc[:, self.X_df_new.columns]
        if self.scaler_X == 'S_Standard' or self.scaler_X == 'U_Standard':
            scaled_X_test = (X_test - self.X_mean) / self.X_std
        else:
            scaled_X_test = X_test
        if self.learning_option == 'PLS':
            pls_X_test = self.pls.transform(scaled_X_test)
            if self.scaler_y == 'S_Standard' or self.scaler_y == 'U_Standard':
                y_pred = self.created_model.predict(pls_X_test) * self.y_std + self.y_mean
            else:
                y_pred = self.created_model.predict(pls_X_test)
        else:
            if self.scaler_y == 'S_Standard' or self.scaler_y == 'U_Standard':
                y_pred = self.created_model.predict(scaled_X_test) * self.y_std + self.y_mean
            else:
                y_pred = self.created_model.predict(scaled_X_test)
        t2 = time.time()
        self.pred_time = t2 - t1
        print(f'Predicted!, Runtime: {t2 - t1} seconds')
        self.reports_li.append((f'Predicted!, Runtime: {t2 - t1} seconds'))
        self.reports_li.append('')
        self.X_test_li = list(X_test)
        self.y_pred_li = list(y_pred.reshape(-1))
        y_pred_df = pd.DataFrame(y_pred.reshape(-1), columns = ['predicted'])
        results_df = pd.concat([y_pred_df, X_test_df], axis = 1)
        # 日付
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        self.date_str = (f'{now:%Y%m%d}')
        if self.learning_option == 'SMOTER':
            csv_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_SMOTER_{self.cleansing_method}_Predicted_{key_word}.csv'
        elif self.learning_option == 'SMOTE':
            csv_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_SMOTE_{self.cleansing_method}_Predicted_{key_word}.csv'
        elif self.learning_option == 'PLS':
            csv_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_PLS_{self.cleansing_method}_Predicted_{key_word}.csv'
        else:
            csv_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{self.cleansing_method}_Predicted_{key_word}.csv'
        results_df.to_csv(f'../results/CSVs/{csv_name}')
        self.results_df = results_df
        self.reports_li.append(f'CSV File with Predicted Values: /results/CSVs/{csv_name}')
        self.reports_li.append('')
        return

    def ScoreGen(self, y_test_df):
        self.y_df = y_test_df
        self.y_test_li = list(y_test_df.values.reshape(-1))
        self.reports_li.append('####### Scores #######')
        r2 = r2_score(self.y_test_li, self.y_pred_li)
        mae = mean_absolute_error(self.y_test_li, self.y_pred_li)
        mse = mean_squared_error(self.y_test_li, self.y_pred_li)
        print(f"R2 Score: {round(r2, 3)}")
        print(f"Mean Absolute Error: {round(mae, 3)}")
        print(f"Mean Squared Error: {round(mse, 3)}")
        self.reports_li.append(f"R2 Score: {round(r2, 3)}")
        self.reports_li.append(f"Mean Absolute Error: {round(mae, 3)}")
        self.reports_li.append(f"Mean Squared Error: {round(mse, 3)}")
        self.reports_li.append("")
        return

    def YYPlotGen(self, key_word='NoKeyWord'):
        # 日付
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        self.date_str = (f'{now:%Y%m%d}')
        if self.learning_option == 'SMOTER':
            figure_png_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_SMOTER_{self.cleansing_method}_YYPlot_CreatedModel_{key_word}.png'
        elif self.learning_option == 'SMOTE':
            figure_png_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_SMOTE_{self.cleansing_method}_YYPlot_CreatedModel_{key_word}.png'
        elif self.learning_option == 'PLS':
            figure_png_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_PLS_{self.cleansing_method}_YYPlot_CreatedModel_{key_word}.png'
        else:
            figure_png_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{self.cleansing_method}_YYPlot_CreatedModel_{key_word}.png'
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test_li, self.y_pred_li, color='blue', s=20)
        plt.plot([min(list(self.y_test_li) + list(self.y_pred_li)), max(list(self.y_test_li) + list(self.y_pred_li))], [min(list(self.y_test_li) + list(self.y_test_li)), max(list(self.y_test_li) + list(self.y_pred_li))], 'r--', lw=1)
        plt.xlabel(f'Actual {self.y_df.columns.values}')
        plt.ylabel(f'Predicted {self.y_df.columns.values}')
        plt.title(f'Predicted vs Actual')
        plt.savefig(f'../results/figures/{figure_png_name}', format="png", transparent = True, dpi = 300)
        plt.show()
        print('The figure is generated!!')
        self.reports_li.append(f'YYPlot: /results/figures/{figure_png_name}')
        self.reports_li.append('')
        return 
    
    def ReportGen(self, key_word='NoKeyWord'):
        # 日付
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        self.date_str = (f'{now:%Y%m%d}')
        if self.learning_option == 'SMOTER':
            text_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_SMOTER_{self.cleansing_method}_Report_CreatedModel_{key_word}.txt'
        elif self.learning_option == 'SMOTE':
            text_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_SMOTE_{self.cleansing_method}_Report_CreatedModel_{key_word}.txt'
        elif self.learning_option == 'PLS':
            text_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_PLS_{self.cleansing_method}_Report_CreatedModel_{key_word}.txt'
        else:
            text_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{self.cleansing_method}_Report_CreatedModel_{key_word}.txt'
        f = open(f'../results/reports/{text_name}', 'w')
        for n in self.reports_li:
            print(n, file = f)
        f.close()
        print('The report is generated!!')
        return
    
    def PreserveInstance(self, key_word='NoKeyWord'):
        # 日付
        t_delta = datetime.timedelta(hours=9)
        JST = datetime.timezone(t_delta, 'JST')
        now = datetime.datetime.now(JST)
        self.date_str = (f'{now:%Y%m%d}')
        if self.learning_option == 'SMOTER':
            instance_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_SMOTER_{self.cleansing_method}_Instance_{key_word}.pkl'
        elif self.learning_option == 'SMOTE':
            instance_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_SMOTE_{self.cleansing_method}_Instance_{key_word}.pkl'
        elif self.learning_option == 'PLS':
            instance_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_PLS_{self.cleansing_method}_Instance_{key_word}.pkl'
        else:
            instance_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{self.cleansing_method}_Instance_{key_word}.pkl'
        pickle.dump(self, open(f'../preserved_instances/{instance_name}', 'wb'))
        print('Reserved!!')
        return
                    
        



