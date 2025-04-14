import yaml
import sys
import numpy as np
sys.path.append('../')

from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import lightgbm as lgb
 

def model_param_grid_selection(selected_model):
    with open('../config/config.yaml', 'r', encoding = 'utf-8') as file:
        config = yaml.safe_load(file)
    param_search = config['learning']['param_search']
    
    # Ridge 
    if selected_model == 'Ridge':
        model = Ridge()
        if config['params']['Ridge']['alpha']['linspace_or_logspace'] == 'linspace':
            start = config['params']['Ridge']['alpha']['start']
            stop = config['params']['Ridge']['alpha']['stop']
            num = config['params']['Ridge']['alpha']['num_or_base']
            param_li = np.linspace(start, stop, num)
        elif config['params']['Ridge']['alpha']['linspace_or_logspace'] == 'logspace':
            start = config['params']['Ridge']['alpha']['start']
            stop = config['params']['Ridge']['alpha']['stop']
            num = stop - start + 1
            base = config['params']['Ridge']['alpha']['num_or_base']
            param_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in linspace_or_logspace")
        if param_search == 'Grid' or param_search == 'Bayes':
            param_grid = {'alpha': param_li}
        else:
            raise ValueError(f"Error in param_search")
        return model, param_grid
    
    # Lasso    
    elif selected_model == 'Lasso':
        model = Lasso(max_iter=10000)
        if config['params']['Lasso']['alpha']['linspace_or_logspace'] == 'linspace':
            start = config['params']['Lasso']['alpha']['start']
            stop = config['params']['Lasso']['alpha']['stop']
            num = config['params']['Lasso']['alpha']['num_or_base']
            alpha_li = np.linspace(start, stop, num)
        elif config['params']['Lasso']['alpha']['linspace_or_logspace'] == 'logspace':
            start = config['params']['Lasso']['alpha']['start']
            stop = config['params']['Lasso']['alpha']['stop']
            num = stop - start + 1
            base = config['params']['Lasso']['alpha']['num_or_base']
            alpha_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in alpha, lasso")
        if param_search == 'Grid' or param_search == 'Bayes':
            param_grid = {'alpha': alpha_li}
        else:
            raise ValueError(f"Error in param_search")
        return model, param_grid
    
    # ElasticNet
    elif selected_model == 'ElasticNet':
        model = ElasticNet(max_iter=10000)
        if config['params']['ElasticNet']['alpha']['linspace_or_logspace'] == 'linspace':
            start = config['params']['ElasticNet']['alpha']['start']
            stop = config['params']['ElasticNet']['alpha']['stop']
            num = config['params']['ElasticNet']['alpha']['num_or_base']
            alpha_li = np.linspace(start, stop, num)
        elif config['params']['ElasticNet']['alpha']['linspace_or_logspace'] == 'logspace':
            start = config['params']['ElasticNet']['alpha']['start']
            stop = config['params']['ElasticNet']['alpha']['stop']
            num = stop - start + 1
            base = config['params']['ElasticNet']['alpha']['num_or_base']
            alpha_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in alpha, ElasticNet")
        if config['params']['ElasticNet']['l1_ratio']['linspace_or_logspace'] == 'linspace':
            start = config['params']['ElasticNet']['l1_ratio']['start']
            stop = config['params']['ElasticNet']['l1_ratio']['stop']
            num = config['params']['ElasticNet']['l1_ratio']['num_or_base']
            l1_ratio_li = np.linspace(start, stop, num)
        elif config['params']['ElasticNet']['l1_ratio']['linspace_or_logspace'] == 'logspace':
            print('l1_ratioは0から1の間に収まっていますか？')
            start = config['params']['ElasticNet']['l1_ratio']['start']
            stop = config['params']['ElasticNet']['l1_ratio']['stop']
            num = stop - start + 1
            base = config['params']['ElasticNet']['l1_ratio']['num_or_base']
            l1_ratio_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in l1_ratio, ElasticNet")
        if param_search == 'Grid' or param_search == 'Bayes':
            param_grid = {'alpha': alpha_li, 'l1_ratio': l1_ratio_li}
        else:
            raise ValueError(f"Error in param_search")
        return model, param_grid
    
    # PLS
    elif selected_model == 'PLS':
        model = PLSRegression()
        if config['params']['PLS']['n_components']['linspace_or_logspace'] == 'linspace':
            start = config['params']['PLS']['n_components']['start']
            stop = config['params']['PLS']['n_components']['stop']
            num = config['params']['PLS']['n_components']['num_or_base']
            n_components_li = np.linspace(start, stop, num)
        elif config['params']['PLS']['n_components']['linspace_or_logspace'] == 'logspace':
            start = config['params']['PLS']['n_components']['start']
            stop = config['params']['PLS']['n_components']['stop']
            num = stop - start + 1
            base = config['params']['PLS']['n_components']['num_or_base']
            n_components_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in alpha, lasso")
        n_components_li = list(map(int, n_components_li))
        if param_search == 'Grid' or param_search == 'Bayes':
            param_grid = {'n_components': n_components_li}
        else:
            raise ValueError(f"Error in param_search")
        return model, param_grid
    
    # SVR
    elif selected_model == 'SVR':
        model = SVR()
        if config['params']['SVR']['C']['linspace_or_logspace'] == 'linspace':
            start = config['params']['SVR']['C']['start']
            stop = config['params']['SVR']['C']['stop']
            num = config['params']['SVR']['C']['num_or_base']
            C_li = np.linspace(start, stop, num)
        elif config['params']['SVR']['C']['linspace_or_logspace'] == 'logspace':
            start = config['params']['SVR']['C']['start']
            stop = config['params']['SVR']['C']['stop']
            num = stop - start + 1
            base = config['params']['SVR']['C']['num_or_base']
            C_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in C, SVR")
        if config['params']['SVR']['epsilon']['linspace_or_logspace'] == 'linspace':
            start = config['params']['SVR']['epsilon']['start']
            stop = config['params']['SVR']['epsilon']['stop']
            num = config['params']['SVR']['epsilon']['num_or_base']
            epsilon_li = np.linspace(start, stop, num)
        elif config['params']['SVR']['epsilon']['linspace_or_logspace'] == 'logspace':
            start = config['params']['SVR']['epsilon']['start']
            stop = config['params']['SVR']['epsilon']['stop']
            num = stop - start + 1
            base = config['params']['SVR']['epsilon']['num_or_base']
            epsilon_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in epsilon, SVR")
        if param_search == 'Grid' or param_search == 'Bayes':
            param_grid = {'C': C_li, 'epsilon': epsilon_li}
        else:
            raise ValueError(f"Error in param_search")
        return model, param_grid  
    
    # RF (Random Forest) 
    elif selected_model == 'RF':
        model = RandomForestRegressor()
        if config['params']['RF']['n_estimators']['linspace_or_logspace'] == 'linspace':
            start = config['params']['RF']['n_estimators']['start']
            stop = config['params']['RF']['n_estimators']['stop']
            num = config['params']['RF']['n_estimators']['num_or_base']
            n_estimators_li = np.linspace(start, stop, num)
        elif config['params']['RF']['n_estimators']['linspace_or_logspace'] == 'logspace':
            start = config['params']['RF']['n_estimators']['start']
            stop = config['params']['RF']['n_estimators']['stop']
            num = stop - start + 1
            base = config['params']['RF']['n_estimators']['num_or_base']
            n_estimators_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in n_estimators, RF")
        if config['params']['RF']['max_depth']['linspace_or_logspace'] == 'linspace':
            start = config['params']['RF']['max_depth']['start']
            stop = config['params']['RF']['max_depth']['stop']
            num = config['params']['RF']['max_depth']['num_or_base']
            max_depth_li = np.linspace(start, stop, num)
        elif config['params']['RF']['max_depth']['linspace_or_logspace'] == 'logspace':
            start = config['params']['RF']['max_depth']['start']
            stop = config['params']['RF']['max_depth']['stop']
            num = stop - start + 1
            base = config['params']['RF']['max_depth']['num_or_base']
            max_depth_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in max_depth, RF")
        n_estimators_li = list(map(int, n_estimators_li))
        max_depth_li = list(map(int, max_depth_li))
        if param_search == 'Grid' or param_search == 'Bayes':
            param_grid = {'n_estimators': n_estimators_li, 'max_depth': max_depth_li}
        else:
            raise ValueError(f"Error in param_search")
        return model, param_grid
        
    # kNN
    elif selected_model == 'kNN':
        model = KNeighborsRegressor()
        if config['params']['kNN']['n_neighbors']['linspace_or_logspace'] == 'linspace':
            start = config['params']['kNN']['n_neighbors']['start']
            stop = config['params']['kNN']['n_neighbors']['stop']
            num = config['params']['kNN']['n_neighbors']['num_or_base']
            n_neighbors_li = np.linspace(start, stop, num)
        elif config['params']['kNN']['n_neighbors']['linspace_or_logspace'] == 'logspace':
            start = config['params']['kNN']['n_neighbors']['start']
            stop = config['params']['kNN']['n_neighbors']['stop']
            num = stop - start + 1
            base = config['params']['kNN']['n_neighbors']['num_or_base']
            n_neighbors_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in n_neighbors, kNN")
        weights_li = config['params']['kNN']['weights']
        metric_li = config['params']['kNN']['metric']
        n_neighbors_li = list(map(int, n_neighbors_li))
        if param_search == 'Grid' or param_search == 'Bayes':
            param_grid = {'n_neighbors': n_neighbors_li, 'weights': weights_li, 'metric': metric_li}
        else:
            raise ValueError(f"Error in param_search")
        return model, param_grid
    
    # GBR (Gradient Boosting)
    elif selected_model == 'GBR':
        model = GradientBoostingRegressor()
        if config['params']['GBR']['n_estimators']['linspace_or_logspace'] == 'linspace':
            start = config['params']['GBR']['n_estimators']['start']
            stop = config['params']['GBR']['n_estimators']['stop']
            num = config['params']['GBR']['n_estimators']['num_or_base']
            n_estimators_li = np.linspace(start, stop, num)
        elif config['params']['GBR']['n_estimators']['linspace_or_logspace'] == 'logspace':
            start = config['params']['GBR']['n_estimators']['start']
            stop = config['params']['GBR']['n_estimators']['stop']
            num = stop - start + 1
            base = config['params']['GBR']['n_estimators']['num_or_base']
            n_estimators_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in n_estimators, GBR")
        if config['params']['GBR']['learning_rate']['linspace_or_logspace'] == 'linspace':
            start = config['params']['GBR']['learning_rate']['start']
            stop = config['params']['GBR']['learning_rate']['stop']
            num = config['params']['GBR']['learning_rate']['num_or_base']
            learning_rate_li = np.linspace(start, stop, num)
        elif config['params']['GBR']['learning_rate']['linspace_or_logspace'] == 'logspace':
            start = config['params']['GBR']['learning_rate']['start']
            stop = config['params']['GBR']['learning_rate']['stop']
            num = stop - start + 1
            base = config['params']['GBR']['learning_rate']['num_or_base']
            learning_rate_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in learning_rate, GBR")
        if config['params']['GBR']['max_depth']['linspace_or_logspace'] == 'linspace':
            start = config['params']['GBR']['max_depth']['start']
            stop = config['params']['GBR']['max_depth']['stop']
            num = config['params']['GBR']['max_depth']['num_or_base']
            max_depth_li = np.linspace(start, stop, num)
        elif config['params']['GBR']['max_depth']['linspace_or_logspace'] == 'logspace':
            start = config['params']['GBR']['max_depth']['start']
            stop = config['params']['GBR']['max_depth']['stop']
            num = stop - start + 1
            base = config['params']['GBR']['max_depth']['num_or_base']
            max_depth_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in max_depth, GBR")
        n_estimators_li = list(map(int, n_estimators_li))
        max_depth_li = list(map(int, max_depth_li))
        if param_search == 'Grid' or param_search == 'Bayes':
            param_grid = {'n_estimators': n_estimators_li, 'learning_rate': learning_rate_li, 'max_depth': max_depth_li}
        else:
            raise ValueError(f"Error in param_search")
        return model, param_grid
    
    # XG Boost
    elif selected_model == 'XGBoost':
        model = XGBRegressor()
        if config['params']['XGBoost']['n_estimators']['linspace_or_logspace'] == 'linspace':
            start = config['params']['XGBoost']['n_estimators']['start']
            stop = config['params']['XGBoost']['n_estimators']['stop']
            num = config['params']['XGBoost']['n_estimators']['num_or_base']
            n_estimators_li = np.linspace(start, stop, num)
        elif config['params']['XGBoost']['n_estimators']['linspace_or_logspace'] == 'logspace':
            start = config['params']['XGBoost']['n_estimators']['start']
            stop = config['params']['XGBoost']['n_estimators']['stop']
            num = stop - start + 1
            base = config['params']['XGBoost']['n_estimators']['num_or_base']
            n_estimators_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in n_estimators, XGBoost")
        if config['params']['XGBoost']['learning_rate']['linspace_or_logspace'] == 'linspace':
            start = config['params']['XGBoost']['learning_rate']['start']
            stop = config['params']['XGBoost']['learning_rate']['stop']
            num = config['params']['XGBoost']['learning_rate']['num_or_base']
            learning_rate_li = np.linspace(start, stop, num)
        elif config['params']['XGBoost']['learning_rate']['linspace_or_logspace'] == 'logspace':
            start = config['params']['XGBoost']['learning_rate']['start']
            stop = config['params']['XGBoost']['learning_rate']['stop']
            num = stop - start + 1
            base = config['params']['XGBoost']['learning_rate']['num_or_base']
            learning_rate_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in learning_rate, XGBoost")
        if config['params']['XGBoost']['max_depth']['linspace_or_logspace'] == 'linspace':
            start = config['params']['XGBoost']['max_depth']['start']
            stop = config['params']['XGBoost']['max_depth']['stop']
            num = config['params']['XGBoost']['max_depth']['num_or_base']
            max_depth_li = np.linspace(start, stop, num)
        elif config['params']['XGBoost']['max_depth']['linspace_or_logspace'] == 'logspace':
            start = config['params']['XGBoost']['max_depth']['start']
            stop = config['params']['XGBoost']['max_depth']['stop']
            num = stop - start + 1
            base = config['params']['XGBoost']['max_depth']['num_or_base']
            max_depth_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in max_depth, XGBoost")
        if config['params']['XGBoost']['min_child_weight']['linspace_or_logspace'] == 'linspace':
            start = config['params']['XGBoost']['min_child_weight']['start']
            stop = config['params']['XGBoost']['min_child_weight']['stop']
            num = config['params']['XGBoost']['min_child_weight']['num_or_base']
            min_child_weight_li = np.linspace(start, stop, num)
        elif config['params']['XGBoost']['min_child_weight']['linspace_or_logspace'] == 'logspace':
            start = config['params']['XGBoost']['min_child_weight']['start']
            stop = config['params']['XGBoost']['min_child_weight']['stop']
            num = stop - start + 1
            base = config['params']['XGBoost']['min_child_weight']['num_or_base']
            min_child_weight_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in min_child_weight, XGBoost")
        if config['params']['XGBoost']['subsample']['linspace_or_logspace'] == 'linspace':
            start = config['params']['XGBoost']['subsample']['start']
            stop = config['params']['XGBoost']['subsample']['stop']
            num = config['params']['XGBoost']['subsample']['num_or_base']
            subsample_li = np.linspace(start, stop, num)
        elif config['params']['XGBoost']['subsample']['linspace_or_logspace'] == 'logspace':
            start = config['params']['XGBoost']['subsample']['start']
            stop = config['params']['XGBoost']['subsample']['stop']
            num = stop - start + 1
            base = config['params']['XGBoost']['subsample']['num_or_base']
            subsample_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in subsample, XGBoost")
        n_estimators_li = list(map(int, n_estimators_li))
        max_depth_li = list(map(int, max_depth_li))
        min_child_weight_li = list(map(int, min_child_weight_li))
        if param_search == 'Grid' or param_search == 'Bayes':
            param_grid = {'n_estimators': n_estimators_li, 'learning_rate': learning_rate_li, 'max_depth': max_depth_li, 'min_child_weight': min_child_weight_li, 'subsample': subsample_li}
        else:
            raise ValueError(f"Error in param_search")
        return model, param_grid
    
    # Light GBM
    elif selected_model == 'LightGBM':
        model = lgb.LGBMRegressor(verbosity= -1)
        if config['params']['LightGBM']['n_estimators']['linspace_or_logspace'] == 'linspace':
            start = config['params']['LightGBM']['n_estimators']['start']
            stop = config['params']['LightGBM']['n_estimators']['stop']
            num = config['params']['LightGBM']['n_estimators']['num_or_base']
            n_estimators_li = np.linspace(start, stop, num)
        elif config['params']['LightGBM']['n_estimators']['linspace_or_logspace'] == 'logspace':
            start = config['params']['LightGBM']['n_estimators']['start']
            stop = config['params']['LightGBM']['n_estimators']['stop']
            num = stop - start + 1
            base = config['params']['LightGBM']['n_estimators']['num_or_base']
            n_estimators_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in n_estimators, LightGBM")
        if config['params']['LightGBM']['learning_rate']['linspace_or_logspace'] == 'linspace':
            start = config['params']['LightGBM']['learning_rate']['start']
            stop = config['params']['LightGBM']['learning_rate']['stop']
            num = config['params']['LightGBM']['learning_rate']['num_or_base']
            learning_rate_li = np.linspace(start, stop, num)
        elif config['params']['LightGBM']['learning_rate']['linspace_or_logspace'] == 'logspace':
            start = config['params']['LightGBM']['learning_rate']['start']
            stop = config['params']['LightGBM']['learning_rate']['stop']
            num = stop - start + 1
            base = config['params']['LightGBM']['learning_rate']['num_or_base']
            learning_rate_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in learning_rate, LightGBM")
        if config['params']['LightGBM']['max_depth']['linspace_or_logspace'] == 'linspace':
            start = config['params']['LightGBM']['max_depth']['start']
            stop = config['params']['LightGBM']['max_depth']['stop']
            num = config['params']['LightGBM']['max_depth']['num_or_base']
            max_depth_li = np.linspace(start, stop, num)
        elif config['params']['LightGBM']['max_depth']['linspace_or_logspace'] == 'logspace':
            start = config['params']['LightGBM']['max_depth']['start']
            stop = config['params']['LightGBM']['max_depth']['stop']
            num = stop - start + 1
            base = config['params']['LightGBM']['max_depth']['num_or_base']
            max_depth_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in max_depth, LightGBM")
        if config['params']['LightGBM']['num_leaves']['linspace_or_logspace'] == 'linspace':
            start = config['params']['LightGBM']['num_leaves']['start']
            stop = config['params']['LightGBM']['num_leaves']['stop']
            num = config['params']['LightGBM']['num_leaves']['num_or_base']
            num_leaves_li = np.linspace(start, stop, num)
        elif config['params']['LightGBM']['num_leaves']['linspace_or_logspace'] == 'logspace':
            start = config['params']['LightGBM']['num_leaves']['start']
            stop = config['params']['LightGBM']['num_leaves']['stop']
            num = stop - start + 1
            base = config['params']['LightGBM']['num_leaves']['num_or_base']
            num_leaves_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in num_leaves, LightGBM")
        if config['params']['LightGBM']['subsample']['linspace_or_logspace'] == 'linspace':
            start = config['params']['LightGBM']['subsample']['start']
            stop = config['params']['LightGBM']['subsample']['stop']
            num = config['params']['LightGBM']['subsample']['num_or_base']
            subsample_li = np.linspace(start, stop, num)
        elif config['params']['LightGBM']['subsample']['linspace_or_logspace'] == 'logspace':
            start = config['params']['LightGBM']['subsample']['start']
            stop = config['params']['LightGBM']['subsample']['stop']
            num = stop - start + 1
            base = config['params']['LightGBM']['subsample']['num_or_base']
            subsample_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in subsample, LightGBM")
        if config['params']['LightGBM']['colsample_bytree']['linspace_or_logspace'] == 'linspace':
            start = config['params']['LightGBM']['colsample_bytree']['start']
            stop = config['params']['LightGBM']['colsample_bytree']['stop']
            num = config['params']['LightGBM']['colsample_bytree']['num_or_base']
            colsample_bytree_li = np.linspace(start, stop, num)
        elif config['params']['LightGBM']['colsample_bytree']['linspace_or_logspace'] == 'logspace':
            start = config['params']['LightGBM']['colsample_bytree']['start']
            stop = config['params']['LightGBM']['colsample_bytree']['stop']
            num = stop - start + 1
            base = config['params']['LightGBM']['colsample_bytree']['num_or_base']
            colsample_bytree_li = np.logspace(start, stop, num = num, base = base)
        else:
            raise ValueError(f"Error in subsample, LightGBM")
        n_estimators_li = list(map(int, n_estimators_li))
        max_depth_li = list(map(int, max_depth_li))
        num_leaves_li = list(map(int, num_leaves_li))
        if param_search == 'Grid' or param_search == 'Bayes':
            param_grid = {'n_estimators': n_estimators_li, 'learning_rate': learning_rate_li, 'max_depth': max_depth_li, 'num_leaves': num_leaves_li, 'subsample': subsample_li, 'colsample_bytree': colsample_bytree_li}
        else:
            raise ValueError(f"Error in param_search")
        return model, param_grid
    else:
        raise ValueError('Error in model selection')    
