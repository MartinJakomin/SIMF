import matplotlib.pyplot as plt
import numpy as np
import tables

from simf import *
from simf.common import split_streams, build_test_set, build_sparse_matrix


def pre_train_models(models, relations, streams):
    main_relation = relations[0]
    test_set = build_test_set(streams[0])
    # Iterate through models
    for m_name, m in models.items():
        print(m_name)
        data = [main_relation]
        if "X3" in m_name:
            data = relations
        # Fit models
        m.fit(data=data, verbose=True)
        # Calculate the error (train and test - remaining stream)
        print("Train error: RMSE:%f,  MAE:%f" % m.get_train_error()[main_relation])
        print("Test error: RMSE:%f,  MAE:%f\n" % m.get_test_error(main_relation, test_set)[main_relation])
    print("Initial factorization complete")
    return models


def evaluate_stream(models, relations, streams, update_interval=3600, max_iter=3, max_updates=None):
    r1, r2, r3 = relations
    results = {}
    # Update models with new params
    for n, m in models.items():
        m.max_iter = max_iter
    print("Starting streaming phase\n")
    for k, s in enumerate(streams):
        streams[k] = list([
            [int(x) for x in s[0]],
            [int(x) for x in s[1]],
            [float(x) for x in s[2]],
            [int(x) for x in s[3]]]
        )
    # Sliding windows of stream data - discrete intervals [current_ts, next_ts]
    current_ts = streams[0][3][0]
    next_ts = current_ts + update_interval
    windows = [[] for k in streams]

    i = 0
    # End when main stream runs out of instances
    while len(streams[0][3]) > 0:
        if max_updates and i >= max_updates:
            break
        row, col, val, ts = [], [], [], []
        # Pop elements into the window
        for k, s in enumerate(streams):
            while len(s[3]) > 0 and s[3][0] <= next_ts:
                row.append(s[0].pop(0))
                col.append(s[1].pop(0))
                val.append(s[2].pop(0))
                ts.append(s[3].pop(0))
            windows[k] = [row, col, val, ts]
            row, col, val, ts = [], [], [], []

        # Continue with next window if no updates on main relation
        if float(len(windows[0][0])) < 1:
            current_ts += update_interval
            next_ts += update_interval
            continue
        # Get true values from the main stream
        true_values = np.array(windows[0][2])
        average_rmse = None

        print("Window: ", i, current_ts, next_ts)
        # Loop through all models
        for m_name, m in models.items():
            # Get predictions for the current window
            predictions = m.predict_stream(r1, list(zip(*windows[0][:3])), verbose=False)
            # Get the error of the average predictor (for calculating rrmse)
            if m_name == 'Average':
                average_rmse = m.rmse(true_values, predictions)
            # Get the error for other predictors and calculate the rrmse
            mrmse = m.rmse(true_values, predictions)
            rrmse = mrmse / average_rmse
            print("%s window error: RMSE:%f,  RRMSE:%f" % (m_name, mrmse, rrmse))
            # Append results
            if m_name not in results:
                results[m_name] = ([mrmse], [rrmse])
            else:
                results[m_name][0].append(mrmse)
                results[m_name][1].append(rrmse)

            # Update models
            r1.set_matrix(build_sparse_matrix(windows[0][:3]))
            r2.set_matrix(build_sparse_matrix(windows[1][:3]))
            r3.set_matrix(build_sparse_matrix(windows[2][:3]))
            data = [r1]
            if "X3" in m_name:
                data = [r1, r2, r3]
            m.fit_update(data=data, verbose=False)

        # Update the interval
        current_ts += update_interval
        next_ts += update_interval
        i += 1
        print()
    return results


def plot_results(results, plot_rrmse=True, l=0.99, title='', save=False, ylim=None):
    for name, res in results.items():
        rrmses = res[int(plot_rrmse)]
        preq_errors = []
        si = 0
        ni = 0
        mdata = np.array(rrmses)
        for r in mdata:
            r = float(r)
            si = r + l * si
            ni = 1 + l * ni
            preq_errors.append(si / float(ni))
        plt.plot(range(len(preq_errors)), preq_errors, label=name)
    if ylim:
        plt.ylim(ylim)
    plt.ylabel('Prequential RRMSE')
    plt.xlabel('Instances (Updates)')
    plt.title(title)
    plt.legend()
    if save:
        plt.savefig(title + '.pdf', bbox_inches='tight')
        plt.clf()
    else:
        plt.show()


# Open data
with tables.open_file('data/yelp_users-restaurants.tbl', 'r') as h5_file:
    s1 = h5_file.root.data.read()
with tables.open_file('data/yelp_users-bars.tbl', 'r') as h5_file:
    s2 = h5_file.root.data.read()
with tables.open_file('data/yelp_users-hotels.tbl', 'r') as h5_file:
    s3 = h5_file.root.data.read()

# Create data streams
train_ur, train_ub, train_uh, test_ur, test_ub, test_uh = split_streams([s1, s2, s3], t=1508546800)
test_set = build_test_set(test_ur)

# Create object types and relations
o_users = ObjectType('users', 25)
o_restaurants = ObjectType('restaurants', 10)
o_bars = ObjectType('bars', 10)
o_hotels = ObjectType('hotels', 10)
r_ur = Relation(o_users, o_restaurants, build_sparse_matrix(train_ur), weight=1)
r_ub = Relation(o_users, o_bars, build_sparse_matrix(train_ub), weight=1)
r_uh = Relation(o_users, o_hotels, build_sparse_matrix(train_uh), weight=1)

# Choose models
models = {
    'Average': Average(),
    'RMF': RMF(rank=25, max_iter=20, regularization=0.005, learning_rate=0.008),
    'SIMF': SIMF(max_iter=20, regularization=0.005, learning_rate=0.008),
    'SIMF_X3': SIMF(max_iter=20, regularization=0.005, learning_rate=0.008),
    'SIMF_X3_CB': SIMF(max_iter=20, regularization=0.005, learning_rate=0.008, combine_bias=True),
}

"""
Recommend restaurants
"""
# Initial factorization
models = pre_train_models(models, [r_ur, r_ub, r_uh], [test_ur, test_ub, test_uh])
# Stream evaluation
results = evaluate_stream(models, [r_ur, r_ub, r_uh], [test_ur, test_ub, test_uh])
# Plot prequential RRMSE
plot_results(results, title='Recommending Restaurants')

"""
Recommend bars
"""
# Initial factorization
models = pre_train_models(models, [r_ub, r_uh, r_ur], [test_ub, test_uh, test_ur])
# Stream evaluation
results = evaluate_stream(models, [r_ub, r_uh, r_ur], [test_ub, test_uh, test_ur])
# Plot prequential RRMSE
plot_results(results, title='Recommending Bars')
