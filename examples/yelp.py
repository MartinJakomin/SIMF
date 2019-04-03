import matplotlib.pyplot as plt
import numpy as np
import tables

from simf import *
from simf.common import split_streams, build_test_set, build_sparse_matrix

# Open data
with tables.open_file('data/yelp_users-bars.tbl', 'r') as h5_file:
    s1 = h5_file.root.data.read()
with tables.open_file('data/yelp_users-restaurants.tbl', 'r') as h5_file:
    s2 = h5_file.root.data.read()
with tables.open_file('data/yelp_users-hotels.tbl', 'r') as h5_file:
    s3 = h5_file.root.data.read()

# Create data streams
train1, train2, train3, test1, test2, test3 = split_streams([s1, s2, s3], t=1508546800)
test_set = build_test_set(test1)

# Create object types and relations
o1 = ObjectType('users', 50)
o2 = ObjectType('bars', 10)
o3 = ObjectType('restaurants', 20)
o4 = ObjectType('hotels', 10)
r1 = Relation(o1, o2, build_sparse_matrix(train1), weight=1)
r2 = Relation(o1, o3, build_sparse_matrix(train2), weight=1)
r3 = Relation(o1, o4, build_sparse_matrix(train3), weight=1)

# Choose models
models = {
    'Average': Average(),
    'SIMF': SIMF(),
    'SIMF_X3': SIMF(),
}

# Initial factorization
for m_name, m in models.items():
    print(m_name)
    data = [r1]
    if "X3" in m_name:
        data = [r1, r2, r3]
    m.fit(data=data, verbose=True)
    print("Train error: RMSE:%f,  MAE:%f" % m.get_train_error()[r1])
    print("Test error: RMSE:%f,  MAE:%f\n" % m.get_test_error(r1, test_set)[r1])
print("Initial factorization complete")

# Streaming phase
max_updates = None
miter = 5
update_interval = 3600
streams = [test1, test2, test3]
results = {}
for n, m in models.items():
    m.max_iter = miter
print("Starting streaming phase\n")
for k, s in enumerate(streams):
    streams[k] = list([
        [int(x) for x in s[0]],
        [int(x) for x in s[1]],
        [float(x) for x in s[2]],
        [int(x) for x in s[3]]]
    )
current_ts = streams[0][3][0]
next_ts = current_ts + update_interval
# Sliding windows of stream data
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
    # True values from the main stream
    true_values = np.array(windows[0][2])
    average_rmse = None

    print("Window: ", i, current_ts, next_ts)
    # Loop through all models
    for m_name, m in models.items():
        predictions = m.predict_stream(r1, list(zip(*windows[0][:3])), verbose=False)
        if m_name == 'Average':
            average_rmse = m.rmse(true_values, predictions)
        # Calculate the error
        if len(predictions) < 1 or np.any(np.isnan(predictions)):
            mrmse = None
            rrmse = None
        else:
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

# Plot prequential RRMSE
plot_rrmse = True
l = 0.99
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

plt.ylabel('Prequential RMSE')
plt.xlabel('Instances (Updates)')
plt.legend()
plt.show()
