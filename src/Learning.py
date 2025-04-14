import time
import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
sys.path.append('../')

from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.cross_decomposition import PLSRegression
from skopt import BayesSearchCV
from src.DataCleansing import Cleansing
from src.Models import model_param_grid_selection
from src.OverSampling import SMOTE_gen

class Learning():
    def __init__(self, X_df, y_df, file_name='NoFileName'):
        # 設定の読み込み
        with open('../config/config.yaml', 'r', encoding = 'utf-8') as file:
            self.config = yaml.safe_load(file)
        # ファイル名(拡張子まで)
        self.file_name = file_name
        # 目的変数
        self.y_df = pd.DataFrame(y_df)
        self.scaler_y = self.config['learning']['scaler_y']
        # 説明変数            
        self.X_df = pd.DataFrame(X_df)
        self.cleansing_method = self.config['feature_engineering']['cleansing']['method']
        self.X_df_new, self.del_variable_numbers_li = self.X_df, []
        self.X_df_new, self.del_variable_numbers_li = Cleansing(self.X_df, self.y_df)
        self.scaler_X = self.config['learning']['scaler_X']
        self.X_std = -10000
        self.X_mean = -10000
        # 学習条件
        self.loss_function = self.config['learning']['loss_function']
        self.method_of_cross_valid = self.config['learning']['cross_validation']
        self.outer_cv = self.config['learning']['DCV'][0]
        self.inner_cv = self.config['learning']['DCV'][1]
        self.random_state = self.config['learning']['random_state']
        self.param_search_method = self.config['learning']['param_search']
        self.learning_option = self.config['learning']['option']
        self.PLS_components = -1
        self.PCA_components = -1
        # スコア
        self.r2 = -10000
        self.mae = -10000
        self.mse = -10000
        # 結果を入れるリスト
        self.X_test_li = []
        self.y_pred_li = []
        self.y_test_li = []
        self.best_params_li = []
        self.reports_li = [] # レポート作成用文字列
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
            threshold_of_Val = self.config['feature_engineering']['cleansing']['same_value_threshold']
            self.reports_li.append(f'                  └── Threshold of Val: {threshold_of_Val}')
        elif self.cleansing_method == 'Val_Corr':
            threshold_of_Val = self.config['feature_engineering']['cleansing']['same_value_threshold']
            threshold_of_Corr = self.config['feature_engineering']['cleansing']['correlation_coefficient_threshold']
            self.reports_li.append(f'                  └── Threshold of Val: {threshold_of_Val}, Threshold of Corr: {threshold_of_Corr}')
        elif self.cleansing_method == 'Chi_Square':
            threshold_of_p_value = self.config['feature_engineering']['cleansing']['p_value_of_chi_square']
            self.reports_li.append(f'                  └── Threshold of p value : {threshold_of_p_value}')
        elif self.cleansing_method == 'ANOVA':
            threshold_of_p_value = self.config['feature_engineering']['cleansing']['p_value_of_ANOVA']
            self.reports_li.append(f'                  └── Threshold of p value : {threshold_of_p_value}')
        elif self.cleansing_method == 'Stepwise_OLS':
            threshold_of_p_value_in = self.config['feature_engineering']['cleansing']['p_value_of_OLS_in']
            threshold_of_p_value_out = self.config['feature_engineering']['cleansing']['p_value_of_OLS_out']
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
        if self.method_of_cross_valid == 'DCV': # Nested Cross Validation
            if self.outer_cv == 'LOO':
                outer_cv_splitter = KFold(n_splits=len(self.X_df_new), shuffle=True, random_state = self.random_state)
                if self.inner_cv == 'LOO':
                    inner_cv_splitter = KFold(n_splits=len(self.X_df_new), shuffle=True, random_state = self.random_state)
                elif self.inner_cv == 'k-Fold' or self.inner_cv == 'k-Fold_2':
                    n_splits = self.config['learning'][self.inner_cv] # k-Fold or k-Fole_2
                    inner_cv_splitter = KFold(n_splits = n_splits, shuffle=True, random_state = self.random_state)
                else:
                    raise ValueError(f"Error in inner CV")        
            elif self.outer_cv == 'k-Fold' or self.outer_cv == 'k-Fold_2':
                n_splits = self.config['learning'][self.outer_cv] # k-Fold or k-Fole_2
                outer_cv_splitter = KFold(n_splits = n_splits, shuffle=True, random_state = self.random_state)
                if self.inner_cv == 'LOO':
                    inner_cv_splitter = KFold(n_splits=len(self.X_df_new), shuffle=True, random_state = self.random_state)
                elif self.inner_cv == 'k-Fold' or self.inner_cv == 'k-Fold_2':
                    inner_n_splits = self.config['learning'][self.inner_cv] # k-Fold or k-Fole_2
                    inner_cv_splitter = KFold(n_splits = inner_n_splits, shuffle=True, random_state = self.random_state)
                else:
                    raise ValueError(f"Error in inner CV")  
            else:
                raise ValueError(f"Error in outer CV")
        else:
            raise ValueError('まだダブルクロスバリデーション(Nested Cross Validation)しか対応しておりません。')
        outer_cv_counter = 0
        t1 = time.time()
        scaled_X = scaled_X.values
        y = self.y_df.values
        # SMOTE
        if self.learning_option == 'SMOTE':
            X_resampled_df, y_resampled_df = SMOTE_gen(pd.concat([self.X_df_new, self.y_df], axis = 1), self.y_df.columns.values[0])
            self.SMOTE_df = pd.concat([X_resampled_df, y_resampled_df], axis = 1)
            # もとのデータは除く
            X_resampled_df = X_resampled_df.iloc[len(self.X_df_new):]
            y_resampled_df = y_resampled_df.iloc[len(self.X_df_new):]
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
            a, b = self.config['learning']['SMOTE'], self.config['learning']['SMOTE_threshold']
            self.reports_li.append(f'Method: {a} SMOTE, Threshold: {b}')
            self.reports_li.append('')
            print('SMOTE Done!')
            if self.scaler_X == 'S_Standard' or self.scaler_X == 'U_Standard':
                scaled_X_resampled = (X_resampled_df - self.X_mean) / self.X_std
            else:
                scaled_X_resampled = X_resampled_df
            scaled_X_resampled = scaled_X_resampled.values
            y_resampled = y_resampled_df.values.reshape(-1)
        
        for train_index, test_index in outer_cv_splitter.split(scaled_X):
            outer_cv_counter += 1
            print(f"------- Outer CV {outer_cv_counter} ----------------------------------------------------------")
            self.reports_li.append(f"------- Outer CV {outer_cv_counter} ----------------------------------------------------------")
            scaled_X_train, scaled_X_test = scaled_X[train_index], scaled_X[test_index]
            y_train, y_test = y[train_index].reshape(-1), y[test_index].reshape(-1)
            # SMOTE
            if self.learning_option == 'SMOTE':
                scaled_X_train = np.concatenate([scaled_X_train, scaled_X_resampled])
                y_train = np.concatenate([y_train, y_resampled])
            Y_mean_tmp = y_train.mean()
            if self.scaler_y == 'S_Standard':
                print('!!標本標準偏差を用いています（不偏標準偏差を使うべきではないですか？）')
                Y_std_tmp = y_train.std(ddof=0)
                scaled_y_train = (y_train - Y_mean_tmp) / Y_std_tmp
            elif self.scaler_y == 'U_Standard':
                Y_std_tmp = y_train.std(ddof=1)
                scaled_y_train = (y_train - Y_mean_tmp) / Y_std_tmp
            else:
                print('Y 標準化なし')
                scaled_y_train = y_train
            if self.learning_option == 'PLS':
                pls = PLSRegression(n_components = self.config['learning']['PLS_components'])
                pls.fit(scaled_X_train, scaled_y_train)
                pls_X_train = pls.transform(scaled_X_train)
            if self.param_search_method == 'Grid':
                param_search_cv = GridSearchCV(estimator = self.model, param_grid = self.params, cv = inner_cv_splitter, scoring = scoring, n_jobs=-1)
            elif self.param_search_method == 'Bayes':
                n_iter = self.config['learning']['Bayes_n_iter']
                param_search_cv = BayesSearchCV(estimator = self.model, search_spaces = self.params, cv = inner_cv_splitter, scoring = scoring, n_iter = n_iter, n_jobs=-1)
            else:
                raise ValueError(f"Error in param_search")
            if self.learning_option == 'PLS':
                param_search_cv.fit(pls_X_train, scaled_y_train)
                df_param_search_cv_tmp = pd.DataFrame(param_search_cv.cv_results_)
                df_param_search_cv = df_param_search_cv_tmp.sort_values('rank_test_score', ascending = True)
                self.reports_li.append(df_param_search_cv.loc[:,['rank_test_score','params','mean_test_score']].head(3))
                self.reports_li.append('')
                print(param_search_cv.best_params_)
                self.reports_li.append(param_search_cv.best_params_)
                best_model = param_search_cv.best_estimator_
                best_model.fit(pls_X_train, scaled_y_train)
                pls_X_test = pls.transform(scaled_X_test)
                if self.scaler_y == 'S_Standard' or self.scaler_y == 'U_Standard':
                    y_pred = best_model.predict(pls_X_test) * Y_std_tmp + Y_mean_tmp
                else:
                    y_pred = best_model.predict(pls_X_test)
            else:
                param_search_cv.fit(scaled_X_train, scaled_y_train)
                df_param_search_cv_tmp = pd.DataFrame(param_search_cv.cv_results_)
                df_param_search_cv = df_param_search_cv_tmp.sort_values('rank_test_score', ascending = True)
                self.reports_li.append(df_param_search_cv.loc[:,['rank_test_score','params','mean_test_score']].head(3))
                self.reports_li.append('')
                print(param_search_cv.best_params_)
                self.reports_li.append(param_search_cv.best_params_)
                best_model = param_search_cv.best_estimator_
                best_model.fit(scaled_X_train, scaled_y_train)
                if self.scaler_y == 'S_Standard' or self.scaler_y == 'U_Standard':
                    y_pred = best_model.predict(scaled_X_test) * Y_std_tmp + Y_mean_tmp
                else:
                    y_pred = best_model.predict(scaled_X_test)
            self.y_pred_li.extend(list(y_pred))
            self.y_test_li.extend(list(y_test))
            for idx in test_index:
                self.X_test_li.append(list(self.X_df.iloc[idx]))    
            self.best_params_li.append(param_search_cv.best_params_)
            print(f"y_test: {y_test}")
            print(f"y_pred: {y_pred}")
            self.reports_li.append(f"y_test: {y_test}, y_pred: {y_pred}")
            tn = time.time()
            print('Cumulative Time: {:.2f} seconds'.format(tn - t1))
            print()
            self.reports_li.append('Cumulative Time: {:.2f} seconds'.format(tn - t1))
            self.reports_li.append('')
        self.reports_li.append('======= End of Learning =====================================================')
        self.reports_li.append('')
        # リスト形式にする
        if type(self.y_pred_li[0]) == np.ndarray or type(self.y_pred_li[0]) == list:
            # 小数点以下5桁にする!!
            self.y_pred_li_new = [round(item[0], 3) for item in self.y_pred_li]
        else:
            self.y_pred_li_new = [round(item, 3) for item in self.y_pred_li]
        if type(self.y_test_li[0]) == np.ndarray or type(self.y_pred_li[0]) == list:
            self.y_test_li_new = [item[0] for item in self.y_test_li]
        else:
            self.y_test_li_new = self.y_test_li
        return
    
    def ResultCSVGen(self, key_word = 'None'):
        y_tmp = pd.DataFrame(zip(self.y_pred_li_new, self.y_test_li_new), columns=[f"predicted {self.y_df.columns.values}", f"actual {self.y_df.columns.values}"])
        X_tmp = pd.DataFrame(self.X_test_li, columns = self.X_df.columns.values)
        results_df = pd.concat([y_tmp, X_tmp], axis = 1)
        if self.learning_option == 'SMOTER':
            csv_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{key_word}_SMOTER_{self.cleansing_method}_ActuralPredicted.csv'
        elif self.learning_option == 'SMOTE':
            csv_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{key_word}_SMOTE_{self.cleansing_method}_ActuralPredicted.csv'
        elif self.learning_option == 'PLS':
            csv_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{key_word}_PLS_{self.cleansing_method}_ActuralPredicted.csv'
        else:
            csv_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{key_word}_{self.cleansing_method}_ActuralPredicted.csv'
        results_df.to_csv(f'../results/CSVs/{csv_name}')
        self.reports_li.append(f'CSV File with Actural and Predicted Values: /results/CSVs/{csv_name}')
        self.reports_li.append('')
        return

    def ScoreGen(self):
        self.reports_li.append('####### Scores #######')
        self.r2 = r2_score(self.y_test_li_new, self.y_pred_li_new)
        self.mae = mean_absolute_error(self.y_test_li_new, self.y_pred_li_new)
        self.mse = mean_squared_error(self.y_test_li_new, self.y_pred_li_new)
        print(f"R2 Score: {round(self.r2, 3)}")
        print(f"Mean Absolute Error: {round(self.mae, 3)}")
        print(f"Mean Squared Error: {round(self.mse, 3)}")
        self.reports_li.append(f"R2 Score: {round(self.r2, 3)}")
        self.reports_li.append(f"Mean Absolute Error: {round(self.mae, 3)}")
        self.reports_li.append(f"Mean Squared Error: {round(self.mse, 3)}")
        self.reports_li.append("")
        return

    def YYPlotGen(self, key_word = 'None'):
        if self.learning_option == 'SMOTER':
            figure_png_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{key_word}_SMOTER_{self.cleansing_method}_YYPlot.png'
        elif self.learning_option == 'SMOTE':
            figure_png_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{key_word}_SMOTE_{self.cleansing_method}_YYPlot.png'
        elif self.learning_option == 'PLS':
            figure_png_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{key_word}_PLS_{self.cleansing_method}_YYPlot.png'
        else:
            figure_png_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{key_word}_{self.cleansing_method}_YYPlot.png'
        plt.figure(figsize=(8, 6))
        plt.scatter(self.y_test_li_new, self.y_pred_li_new, color='blue', s=20)
        plt.plot([min(list(self.y_test_li_new) + list(self.y_pred_li_new)), max(list(self.y_test_li_new) + list(self.y_pred_li_new))], [min(list(self.y_test_li_new) + list(self.y_test_li_new)), max(list(self.y_test_li_new) + list(self.y_pred_li_new))], 'r--', lw=1)
        plt.xlabel(f'Actual {self.y_df.columns.values}')
        plt.ylabel(f'Predicted {self.y_df.columns.values}')
        plt.title(f'Predicted vs Actual')
        plt.savefig(f'../results/figures/{figure_png_name}', format="png", transparent = True, dpi = 300)
        plt.show()
        print('The figure is generated!!')
        self.reports_li.append(f'YYPlot: /results/figures/{figure_png_name}')
        self.reports_li.append('')
        return 
    
    def ReportGen(self, key_word = 'None'):
        if self.learning_option == 'SMOTER':
            text_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{key_word}_SMOTER_{self.cleansing_method}_Report.txt'
        elif self.learning_option == 'SMOTE':
            text_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{key_word}_SMOTE_{self.cleansing_method}_Report.txt'
        elif self.learning_option == 'PLS':
            text_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{key_word}_PLS_{self.cleansing_method}_Report.txt'
        else:
            text_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_{key_word}_{self.cleansing_method}_Report.txt'
        f = open(f'../results/reports/{text_name}', 'w')
        for n in self.reports_li:
            print(n, file = f)
        f.close()
        print('The report is generated!!')
        return 
    
    