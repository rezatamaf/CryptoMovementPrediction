from functools import partial
from dateutil.relativedelta import *
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import optuna
from sklearn.model_selection import train_test_split


class TimeBasedCV(object):
    """
    Parameters
    ----------
    train_period: int
        number of time units to include in each train set
        default is 30
    test_period: int
        number of time units to include in each test set
        default is 7
    freq: string
        frequency of input parameters. possible values are: days, months, years, weeks, hours, minutes, seconds
        possible values designed to be used by dateutil.relativedelta class
        default is days
    """

    def __init__(self, train_period=30, test_period=7, freq='days'):
        self.n_splits = None
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq

    def split(self, data, validation_split_date=None, date_column='record_date', gap=0):
        """
        Generate indices to split data into training and test set

        Parameters
        ----------
        data: pandas DataFrame
            your data, contain one column for the record date
        validation_split_date: datetime.date()
            first date to perform the splitting on.
            if not provided will set to be the minimum date in the data after the first training set
        date_column: string, default='record_date'
            date of each record
        gap: int, default=0
            for cases the test set does not come right after the train set,
            *gap* days are left between train and test sets

        Returns
        -------
        train_index ,test_index:
            list of tuples (train index, test index) similar to sklearn model selection
        """

        # check that date_column exist in the data:
        try:
            data[date_column]
        except ValueError:
            raise KeyError(date_column)

        train_indices_list = []
        test_indices_list = []

        if validation_split_date is None:
            validation_split_date = data[date_column].min().date() \
                                    + eval('relativedelta( ' + self.freq + '=self.train_period)')

        start_train = validation_split_date - eval('relativedelta( ' + self.freq + '=self.train_period)')
        end_train = start_train + eval('relativedelta( ' + self.freq + '=self.train_period)')
        start_test = end_train + eval('relativedelta( ' + self.freq + '=gap)')
        end_test = start_test + eval('relativedelta( ' + self.freq + '=self.test_period)')

        while start_test <= data[date_column].max().date():
            # train indices:
            cur_train_indices = list(data[(data[date_column].dt.date >= start_train) &
                                          (data[date_column].dt.date < end_train)].index)

            # test indices:
            cur_test_indices = list(data[(data[date_column].dt.date >= start_test) &
                                         (data[date_column].dt.date < end_test)].index)

            train_indices_list.append(cur_train_indices)
            test_indices_list.append(cur_test_indices)

            # update dates:
            start_train = start_train + eval('relativedelta( ' + self.freq + '=self.test_period)')
            end_train = start_train + eval('relativedelta( ' + self.freq + '=self.train_period)')
            start_test = end_train + eval('relativedelta( ' + self.freq + '=gap)')
            end_test = start_test + eval('relativedelta( ' + self.freq + '=self.test_period)')

        # mimic sklearn output
        index_output = [(train, test) for train, test in zip(train_indices_list, test_indices_list)]

        self.n_splits = len(index_output)

        return index_output

    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits


def cv_train(X, y, cv, date_column, validation_split_date, clf, clf_params=None):
    preds = np.array([])
    actuals = np.array([])
    for train_index, test_index in cv.split(X, validation_split_date, date_column):
        data_train = X.loc[train_index].drop(date_column, axis=1)
        target_train = y.loc[train_index]

        data_test = X.loc[test_index].drop(date_column, axis=1)
        target_test = y.loc[test_index]

        if clf_params is not None:
            model = clf(random_state=42, **clf_params)
        else:
            model = clf(random_state=42)
        model.fit(data_train, target_train)

        pred = model.predict_proba(data_test)

        preds = np.append(preds, pred[:, 1])
        actuals = np.append(actuals, target_test.values)
    return preds, actuals


def split_to_window(source_list, window_size):
    return [source_list[x:x + window_size] for x in range(0, len(source_list), window_size)]


def evaluate_on_window(predictions, actuals, threshold=0.5, window_size=7):
    pred_classes = [i > threshold for i in predictions]
    actuals_chunk = split_to_window(actuals, window_size)
    preds_chunk = split_to_window(pred_classes, window_size)
    rolling_acc = []
    for actual, pred in zip(actuals_chunk, preds_chunk):
        rolling_acc.append(accuracy_score(actual, pred))
    mean_acc = np.mean(rolling_acc)
    std_acc = np.std(rolling_acc)
    print(f'{window_size}-days window accuracy avg: {np.round(mean_acc, 2)}')
    print(f'{window_size}-days window accuracy std: {np.round(std_acc, 2)}\n')
    return mean_acc, std_acc


def hpo_objective(trial, clf, X, y, date_column, split_date, cv):
    # Optuna suggest params
    params = {
        "objective": "binary:logistic",
        "tree_method": "exact",
        "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 350, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.01, 0.10),
        'subsample': trial.suggest_uniform('subsample', 0.50, 0.90),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.50, 0.90),
        'gamma': trial.suggest_int('gamma', 0, 20),
        'missing': -999
    }

    preds, actuals = cv_train(X, y, cv, date_column, split_date, clf, clf_params=params)
    pred_class = [j > 0.5 for j in preds]
    return f1_score(actuals.astype(bool), pred_class, average='macro')


def run_hpo(clf, X, y, date_column, split_date, cv, n_trials):
    obj_fun = partial(hpo_objective, clf=clf, X=X, y=y, date_column=date_column, split_date=split_date, cv=cv)
    study = optuna.create_study(direction="maximize")
    study.optimize(obj_fun, n_trials=n_trials)
    print("Number of finished trials: {}".format(len(study.trials)))
    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    print("")
    return trial


def fine_tune_model(uninit_clf, X, y, train_period, holdout_period=14, date_column='date', n_trials=20):
    hpo_split_date = X.index[train_period].date()
    holdout_split_date = X.index[-holdout_period].date()
    X.index.name = date_column
    X = X.reset_index()
    y = y.reset_index(drop=True)
    X_train, _, y_train, _ = train_test_split(X, y, test_size=holdout_period, shuffle=False)
    tscv = TimeBasedCV(train_period, test_period=1, freq='days')
    trial = run_hpo(uninit_clf, X_train, y_train, date_column, hpo_split_date, tscv, n_trials)
    data_preds, data_actuals = cv_train(X_train, y_train, tscv, date_column, hpo_split_date,
                                        uninit_clf, clf_params=trial.params)
    mean_acc, std_acc = evaluate_on_window(data_preds, data_actuals, window_size=7)
    # eval on holdout
    holdout_preds, holdout_actuals = cv_train(X, y, tscv, date_column, holdout_split_date,
                                              uninit_clf, clf_params=trial.params)

    holdout_mean_acc, holdout_std_acc = evaluate_on_window(holdout_preds, holdout_actuals, window_size=7)
    eval_metrics = {'eval_acc': mean_acc, 'eval_acc_7days_std': std_acc,
                    'holdout_eval_acc': holdout_mean_acc, 'holdout_eval_acc_7days_std': holdout_std_acc}
    return trial.params, eval_metrics
