import pandas as pd
import numpy as np

# Load data
df = pd.read_csv('data/votes.csv')
df['date'] = pd.to_datetime(df['date'])

# 1. Clean data: Handle missing parties
# Fill party from same actor by earliest known date
df = df.sort_values(['actor', 'date'])
df['party'] = df.groupby('actor')['party'].ffill().bfill()

# 2. Filter: drop actor 20976
df = df[df['actor'] != 20976]

# 3. Filtering logic
# A MP registered for <50% of sessions
# Total unique sessions
all_sessions = df['session'].unique()
n_total_sessions = len(all_sessions)
actor_sessions = df.groupby('actor')['session'].nunique()
registered_actors = actor_sessions[actor_sessions >= (0.5 * n_total_sessions)].index
df = df[df['actor'].isin(registered_actors)]

# B Drop MPs present (vote in {1,2,4}) for <50% of sessions during their own tenure
# Recalculate after first filter
actor_sessions = df.groupby('actor')['session'].nunique()

# Number of sessions they were present (vote in {1,2,4})
present_votes = df[df['vote'].isin([1, 2, 4])]
actor_present_sessions = present_votes.groupby('actor')['session'].nunique()

# Filter: present for >= 50% of sessions they attended
valid_actors = actor_present_sessions[actor_present_sessions >= (0.5 * actor_sessions)].index
df = df[df['actor'].isin(valid_actors)]

print(f"Retained {df['actor'].nunique()} MPs.")

# 4. Prepare labels
# Define UFG flag
ufg_keywords = ['Uden for folketingsgrupperne']
df['ufg_flag'] = df['party'].apply(lambda x: 1 if any(k in str(x) for k in ufg_keywords) else 0)

# Create label sets
# A: first
first_label = df.sort_values('date').groupby('actor')['party'].first()
# B: modal
modal_label = df.groupby('actor')['party'].apply(lambda x: x.mode()[0])

# C: noswitch
switched = df.groupby('actor')['party'].nunique() > 1
noswitch_actors = switched[~switched].index
noswitch_label = modal_label[noswitch_actors]

# Helper to save
def save_csv(series, filename):
    res = pd.DataFrame({'actor': series.index, 'party_label': series.values})
    res['ufg_flag'] = res['party_label'].apply(lambda x: 1 if any(k in str(x) for k in ufg_keywords) else 0)
    res.to_csv(f'outputs/{filename}.csv', index=False)
    print(f"{filename}: {len(res)} rows")
    print(f"Class distribution:\n{res['party_label'].value_counts()}\n")

save_csv(first_label, 'ground_truth_first')
save_csv(modal_label, 'ground_truth_modal')
save_csv(noswitch_label, 'ground_truth_noswitch')

# Report diffs
diffs = (first_label != modal_label).sum()
print(f"Labels differing between first and modal: {diffs}")
