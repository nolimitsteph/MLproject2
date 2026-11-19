with open('housing.csv') as f:
    for _ in range(3):
        print(repr(f.readline()))

print(dataset.shape)
print(dataset.head(3))
print(dataset['MEDV'].isna().sum(), dataset['MEDV'].nunique())