import pandas as pd
from scipy.spatial.distance import euclidean, pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

INFILE = 'spotify.csv'
N_SAMPLES = 5000


def weighted_euclid(a, b, weights):
    '''
    get the euclidean distance between a and b
    :param a: list. point a. list of numeric coordinates
    :param b: list. point b. list of numeric coordinates
    :return: float. the euclidean distance between a and b
    '''

    # ensure that both points have same number of features
    if len(a) != len(b):
        raise Exception('a and b must have the same number of features.')

    # return euclidean distance between a and b, weighting each feature accordingly
    return sum([(weights[i]*(a[i] - b[i]))**2 for i in range(len(a))])**0.5


def get_sim_df(df, sim_fn, **kwargs):
    '''
    gets a dataframe with pairwise distances between points calculated by a given similarity function
    :param df: dataframe of features - each row represents a point
    :param sim_fn: function to calculate similarity by
    :return: dataframe with similarities given as ['source', 'target', 'similarity']
    '''
    # get similarity matrix
    dists = pdist(df, sim_fn, **kwargs)
    sim_matrix = pd.DataFrame(squareform(dists), columns=df.index, index=df.index)

    # pivot and clean
    sim_matrix = sim_matrix.stack()
    sim_matrix.index.names = ['source', 'target']
    sim_matrix = sim_matrix.reset_index()
    sim_matrix.columns = ['source', 'target', 'similarity']

    return sim_matrix


def get_weights(X, y):
    """
    uses simple linear regression to get weights for a set of features
    :param X: feature data
    :param y: labels
    :return: array of weights corresponding each feature
    """
    reg = LinearRegression()
    reg.fit(X, y)
    return reg.coef_


def main():

    # features to measure 'similarity' based on
    features = ['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness',
                'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature']
    spotify = pd.read_csv(INFILE)

    # scale numerical features
    scaler = StandardScaler()
    spotify[features] = scaler.fit_transform(spotify[features])

    # compute feature weights based on genre classifications
    encoder = LabelEncoder()
    feature_weights = get_weights(spotify[features], encoder.fit_transform(spotify['track_genre']))

    spotify.drop_duplicates(['track_id'], inplace=True)

    # sample spotify data making sure to include all songs from is this it
    strokes = spotify[spotify['album_name'] == 'Is This It'].reset_index(drop=True)
    samples = pd.concat([spotify.sample(N_SAMPLES), strokes]).drop_duplicates().reset_index(drop=True)

    samples.to_csv('~/opt/neo4j-community-5.18.1/import/spotify_samples.csv')

    # get similarity scores for each song pair in sample (smaller distance more similar)
    samples = samples.set_index('track_id')
    song_sims = get_sim_df(samples[features], weighted_euclid, weights=feature_weights)

    # only label songs as "similar" if distance is in the top 1 percentile - ignore score of 0 (same song)
    song_sims = song_sims[(song_sims['similarity'] < song_sims['similarity'].quantile(0.01)) & (song_sims['similarity'] > 0)]

    song_sims.to_csv('~/opt/neo4j-community-5.18.1/import/song_similarities.csv')


if __name__ == "__main__":
    main()
