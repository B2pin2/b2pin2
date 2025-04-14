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
from src.Models import model_param_grid_selection
from src.OverSampling import SMOTE_gen
from src.Learning import Learning

# あまり使わないかな？, CVごとに毎回SMOTEするというのはかなりマイナー
class LearningTrainSMOTE(Learning):
    def __init__(self, X_df, y_df, file_name = 'NoFileName'):
        super(LearningTrainSMOTE, self).__init__(X_df, y_df, file_name)
        return
    # trainデータにのみSMOTEを適用する
    # yamlでSMOTEを指定しなくても大丈夫（PLSなどのoptionは指定してもよい）
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
        y = self.y_df    
        a, b = self.config['learning']['SMOTE'], self.config['learning']['SMOTE_threshold']
        self.reports_li.append(f'Method: {a} SMOTE, Threshold: {b}')
        
        for train_index, test_index in outer_cv_splitter.split(scaled_X):
            outer_cv_counter += 1
            self.reports_li.append(f"------- Outer CV {outer_cv_counter} ----------------------------------------------------------")
            print(f"------- Outer CV {outer_cv_counter} ----------------------------------------------------------")
            scaled_X_train, scaled_X_test = scaled_X.iloc[train_index, :], scaled_X.iloc[test_index, :]
            y_train, y_test = y.iloc[train_index, :], y.iloc[test_index, :]    
            # SMOTE
            X_resampled_df, y_resampled_df = SMOTE_gen(pd.concat([scaled_X_train, y_train], axis = 1), self.y_df.columns.values[0])
            self.SMOTE_df = pd.concat([X_resampled_df, y_resampled_df], axis = 1)
            # もとのデータを含む
            self.reports_li.append(f'======= SMOTE resampled_X ===================================================')
            self.reports_li.append(X_resampled_df)
            self.reports_li.append('')
            self.reports_li.append(f'======= SMOTE resampled_y ===================================================')
            self.reports_li.append(y_resampled_df)
            self.reports_li.append('')
            if self.scaler_X == 'S_Standard' or self.scaler_X == 'U_Standard':
                scaled_X_resampled = (X_resampled_df - self.X_mean) / self.X_std
            else:
                scaled_X_resampled = X_resampled_df
            scaled_X_train, scaled_X_test = scaled_X_resampled.values, scaled_X_test.values
            y_train, y_test = y_resampled_df.values.reshape(-1), y_test.values.reshape(-1)
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
            # 小数点以下一桁にする!!
            self.y_pred_li_new = [round(item[0], 1) for item in self.y_pred_li]
        else:
            self.y_pred_li_new = [round(item, 1) for item in self.y_pred_li]
        if type(self.y_test_li[0]) == np.ndarray or type(self.y_pred_li[0]) == list:
            self.y_test_li_new = [item[0] for item in self.y_test_li]
        else:
            self.y_test_li_new = self.y_test_li
        return
    
    def ResultCSVGen(self):
        y_tmp = pd.DataFrame(zip(self.y_pred_li_new, self.y_test_li_new), columns=[f"predicted {self.y_df.columns.values}", f"actual {self.y_df.columns.values}"])
        X_tmp = pd.DataFrame(self.X_test_li, columns = self.X_df.columns.values)
        results_df = pd.concat([y_tmp, X_tmp], axis = 1)
        if self.learning_option == 'PLS':
            csv_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_TrainSMOTE_PLS_{self.cleansing_method}_ActuralPredicted.csv'
        else:
            csv_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_TrainSMOTE_{self.cleansing_method}_ActuralPredicted.csv'
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

    def YYPlotGen(self):
        if self.learning_option == 'PLS':
            figure_png_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_TrainSMOTE_PLS_{self.cleansing_method}_YYPlot.png'
        else:
            figure_png_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_TrainSMOTE_{self.cleansing_method}_YYPlot.png'
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
    
    def ReportGen(self):
        if self.learning_option == 'PLS':
            text_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_TrainSMOTE_PLS_{self.cleansing_method}_Report.txt'
        else:
            text_name = f'{self.date_str}_{self.file_name[:-4]}_{self.selected_model}_TrainSMOTE_{self.cleansing_method}_Report.txt'
        f = open(f'../results/reports/{text_name}', 'w')
        for n in self.reports_li:
            print(n, file = f)
        f.close()
        print('The report is generated!!')
        return 
    
   