import os
import pandas as pd

data_location = '120 Categorization'

names = {
    'Igneous': ['Andesite', 'Basalt', 'Diorite', 'Gabbro', 'Granite', 'Obsidian', 'Pegmatite', 'Peridotite', 'Pumice',
                'Rhyolite'],
    'Metamorphic': ['Amphibolite', 'Anthracite', 'Gneiss', 'Hornfels', 'Marble', 'Migmatite', 'Phyllite', 'Quartzite',
                    'Schist', 'Slate'],
    'Mixed': ['Basalt', 'Diorite', 'Obsidian', 'Pumice', 'Anthracite', 'Marble', 'Dolomite', 'Micrite', 'Rock Gypsum',
              'Sandstone']}

data = pd.DataFrame()
for subdir, dirs, files in os.walk(data_location):
    for f in files:
        file = os.path.join(subdir, f)
        df = pd.read_csv(file, delim_whitespace=True, index_col=False)
        data = data.append(df)

data['Response'] += 10 * (data['Response'] == 0)
data['Training'] = [row['Token'] < 3 for i, row in data.iterrows()]
data['Cond'] = [['Igneous', 'Metamorphic', 'Sedimentary', 'Mixed'][row['Condition'] - 1] for i, row in data.iterrows()]
data['Condition'] = data['Cond']
data['Category'] = [names[row['Condition']][row['Subtype'] - 1] for i, row in data.iterrows()]

transfer = data.query('Session == 2').groupby(['Condition', 'Subject'], as_index=False).mean()
outliers = transfer.query('Correct < .6')['Subject']
data['outlier'] = [row['Subject'] in list(outliers) for i, row in data.iterrows()]
data['Item_Type'] = ['Training' if row['Token'] < 3 else 'Test' for i, row in data.iterrows()]

data.to_csv('categorization_120_data.txt', sep='\t', index=False)