from joblib import dump, load
import numpy as np
import time
from cuml import ForestInference as FIL

from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import treelite
import treelite.runtime
import xgboost as xgb

from numba import cuda
import cudf

import csv
from pathlib import Path

repeat = 3

def simulate_data(m, n, k=2, random_state=None, classification=True):
    if classification:
        features, labels = make_classification(n_samples=m, n_features=n,
                                                n_informative=int(n/5), n_classes=k,
                                                random_state=random_state, shuffle=False)
    else:
        features, labels = make_regression(n_samples=m, n_features=n,
                                            n_informative=int(n/5), n_targets=1,
                                            random_state=random_state, shuffle=False)
    return features.astype(np.float32), labels.astype(np.float32)

model_path = "/home/rlan/Desktop/RF/benchmarking_fil/cuml/python/cuml/test/"

def train_xgb(max_depth, n_trees, n_cols, X_train, y_train):
    print("===>Training XGB - D: %d, T: %d, C: %d" % (max_depth, n_trees, n_cols))

    if Path(model_path+'xgb_D'+str(max_depth)+'_T'+str(n_trees)+'_C'+str(n_cols)+'.model').is_file():
        print("    Model exist, exiting")
        return 
    
    dtrain = xgb.DMatrix(X_train[:, :n_cols], label=y_train)

    # instantiate params
    params = {}
    # general params
    general_params = {'silent': 1}
    params.update(general_params)
    # learning task params
    learning_task_params = {}
    learning_task_params['eval_metric'] = 'error'
    # predict 0 or 1 instead of probability 
    learning_task_params['objective'] = 'binary:logistic'
    learning_task_params['max_depth'] = max_depth
    learning_task_params['base_score'] = 0.5
    params.update(learning_task_params)    

    start_xgb = time.time()
    xgb_tree = xgb.train(params, dtrain, n_trees)
    stop_xgb = time.time()
    print("    XGboost training time: ", stop_xgb - start_xgb)
    xgb_tree.save_model(model_path+'xgb_D'+str(max_depth)+'_T'+str(n_trees)+'_C'+str(n_cols)+'.model')

def train_skl(max_depth, n_trees, n_cols, X_train, y_train):
    print("===>Training SKL - D: %d, T: %d, C: %d" % (max_depth, n_trees, n_cols))
    skl_tree = RandomForestClassifier(max_depth=max_depth, 
                                        max_features=1.0, 
                                        n_estimators=n_trees)
    
    start_skl = time.time()
    skl_tree.fit(X_train, y_train)
    stop_skl = time.time()
    print("    Sklearn training time: ", stop_skl - start_skl)
    dump(skl_tree, model_path+'skl_tree_D'+str(max_depth)+'_T'+str(n_tree)+'_C'+str(n_cols)+'.joblib') 

def train_all(max_depth, n_trees, n_cols, X_train, y_train):
    train_skl(max_depth, n_trees, n_cols, X_train, y_train)
    train_xgb(max_depth, n_trees, n_cols, X_train, y_train)

def test_all(max_depth, n_trees, n_cols, n_rows, test_models, X_test, y_test):
    print("===>Testing - D: %d, T: %d, C: %d, R: %d" % (max_depth, n_trees, n_cols, n_rows))

    xgb_tree = xgb.Booster()
    xgb_tree.load_model(model_path+'xgb_D'+str(max_depth)+'_T'+str(n_trees)+'_C'+str(n_cols)+'.model') 

    fm = {}
    algos = ['NAIVE', 'TREE_REORG', 'BATCH_TREE_REORG']

    for algo in algos:
        fm[algo] = FIL.load(model_path+'xgb_D'+str(max_depth)+'_T'+str(n_trees)+'_C'+str(n_cols)+'.model',
                            algo=algo, output_class=True, threshold=0.50)

    write_csv = []
    acc_csv = []

    X_test_g = cuda.to_device(np.ascontiguousarray(X_test[:n_rows, :n_cols]))
    X_test_c = X_test[:n_rows, :n_cols]
    y_test = y_test[:n_rows]

    write_csv.append('D'+str(max_depth)+'_T'+str(n_trees)+'_C'+str(n_cols)+'R_'+str(n_rows))

    # Test SKL 
    """
    if 'skl' in test_models:
        for run in range(3):
            each_run = []
            start_skl = time.time()
            skl_preds = skl_tree.predict(X_test[:n_rows, :])
            stop_skl = time.time()
            skl_acc = accuracy_score(skl_preds, y_test[:n_rows])
            print("Sklearn testing time: ", (stop_skl - start_skl) * 1000, " Sklearn acc: ", skl_acc)
            each_run.append((stop_skl - start_skl) * 1000)
            acc_csv.append(skl_acc)
        write_csv.append(min(each_run))
    """

    # Test XGB CPU
    if 'xgb_cpu' in test_models:
        xgb_tree.set_param({'predictor': 'cpu_predictor'})
        dtest = xgb.DMatrix(X_test_c, label=y_test)
        each_run = []        

        for run in range(repeat):

            _ = xgb_tree.predict(dtest)
            start_xgb = time.time()
            xgb_preds_cpu = xgb_tree.predict(dtest)
            stop_xgb = time.time()

            xgb_acc_cpu = accuracy_score(xgb_preds_cpu > 0.5, y_test)
            each_run.append((stop_xgb - start_xgb) * 1000)

        acc_csv.append(xgb_acc_cpu)    
        write_csv.append(min(each_run))
        print("    XGboost CPU testing time: ", min(each_run), " XGboost CPU acc: ", xgb_acc_cpu)

    # Test XGB GPU 
    if 'xgb_gpu' in test_models:
        xgb_tree.set_param({'predictor': 'gpu_predictor'})
        xgb_tree.set_param({'n_gpus': '1'})
        xgb_tree.set_param({'gpu_id': '0'})
        dtest = xgb.DMatrix(X_test_g, label=y_test)
        each_run = []

        for run in range(repeat):

            _ = xgb_tree.predict(dtest)
            start_xgb = time.time()
            xgb_preds_gpu = xgb_tree.predict(dtest)
            stop_xgb = time.time()

            xgb_acc_gpu = accuracy_score(xgb_preds_gpu > 0.5, y_test)
            each_run.append((stop_xgb - start_xgb) * 1000)

        acc_csv.append(xgb_acc_gpu)
        write_csv.append(min(each_run))
        print("    XGboost GPU testing time: ", min(each_run), " XGboost GPU acc: ", xgb_acc_gpu)

    # Test Treelite
    if 'treelite' in test_models:
        tl_model = treelite.Model.from_xgboost(xgb_tree)
        toolchain = 'gcc'
        tl_model.export_lib(toolchain=toolchain, libpath=model_path +'_D'+str(max_depth)+'_T'+str(n_trees)+'_C' + str(n_cols)+'R_'+ str(n_rows)+'.so', params={'parallel_comp': 12}, verbose=False)
        tl_predictor = treelite.runtime.Predictor(model_path +'_D'+str(max_depth)+'_T'+str(n_trees)+'_C' + str(n_cols)+'R_'+ str(n_rows)+'.so', verbose=False)
        tl_batch = treelite.runtime.Batch.from_npy2d(X_test_c)
        each_run = []        

        for run in range(repeat):
            _ = tl_predictor.predict(tl_batch)
            start_tl = time.time()
            tl_pred = tl_predictor.predict(tl_batch)
            stop_tl = time.time()

            tl_acc = accuracy_score(tl_pred > 0.5, y_test)
            each_run.append((stop_tl - start_tl) * 1000)

        acc_csv.append(tl_acc)    
        write_csv.append(min(each_run))
        print("    Treelite CPU testing time: ", min(each_run), " Treelite CPU acc: ", tl_acc)

    if 'fil' in test_models:
        # For testing the effect when using a device array for the inputs. 
        for algo in algos:
            each_run = []
            for run in range(repeat):
                _ = fm[algo].predict(X_test_g)
                start_fil = time.time()
                fil_preds = fm[algo].predict(X_test_g)
                stop_fil = time.time()

                fil_acc = accuracy_score(fil_preds, y_test)
                each_run.append((stop_fil - start_fil) * 1000)

            acc_csv.append(fil_acc)
            write_csv.append(min(each_run))
            print("    FIL %s testing time: " % algo, min(each_run), " FIL %s acc: " % algo, xgb_acc_gpu)

    for elem in acc_csv:
        write_csv.append(elem)

    with open("./result_test_0812.csv", 'a', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(write_csv)

if __name__ == '__main__':

    # control the test cases 
    test_rows = [1000]
    test_cols = [16]
    test_trees = [100]
    test_depth = [6]
    test_models = ['xgb_gpu', 'xgb_cpu', 'fil', 'treelite']

    print("===========================================")
    print("Benchmark Starts")

    for max_depth in test_depth:
        for n_trees in test_trees:
            for n_cols in test_cols:
                dataset_rows = 1100000
                # Generate dataset 
                X, y = simulate_data(dataset_rows, n_cols, 2, random_state=43210, classification=True)
                # identify shape and indices
                # print("X shape: ", X.shape)
                # print("y shape: ", y.shape)

                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=100000, random_state=43210, shuffle=False)

                train_xgb(max_depth, n_trees, n_cols, X_train, y_train)

                for n_rows in test_rows:
                    print("===========================================")
                    # print("X_train shape: ", X_train.shape)
                    # print("y_train shape: ", y_train.shape)
                    # print("X_test shape: ", X_test.shape)
                    # print("y_test shape: ", y_test.shape)

                    test_all(max_depth, n_trees, n_cols, n_rows, test_models, X_test, y_test)